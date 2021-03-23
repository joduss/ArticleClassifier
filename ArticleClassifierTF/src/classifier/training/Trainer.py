import logging
from logging import getLogger
from typing import Any, List, Optional

from tensorflow.python.keras.callbacks import LambdaCallback

from classifier.Data.TrainValidationDataset import TrainValidationDataset
from classifier.evaluation.abstracts.ThemeMetricAggregator import ThemeMetricAggregator
from classifier.prediction.articles_prediction import ArticlesPrediction
from classifier.prediction.article_predictor import ArticlePredictor
from classifier.models.IClassifierModel import IClassifierModel
from classifier.preprocessing.article_text_tokenizer import ArticleTextTokenizer
from classifier.preprocessing.article_theme_tokenizer import ArticleThemeTokenizer
from classifier.preprocessing.interface_article_preprocessor import IArticlePreprocessor
from classifier.training.TrainedModel import TrainedModel
from data_models.ThemeStat import ThemeStat
from data_models.articles import Articles
from data_models.weights.theme_weights import ThemeWeights


class Trainer(LambdaCallback):

    logger = getLogger("PN")

    _themes: List[str]

    _preprocessor: IArticlePreprocessor
    _processed_articles: Articles
    _max_article_length: int

    _article_tokenizer: ArticleTextTokenizer
    _theme_tokenizer: ArticleThemeTokenizer

    _theme_stats: List[ThemeStat] = []
    _theme_metrics__: List[ThemeMetricAggregator]

    _trained: bool = False

    validation_ratio: float = 0.2
    batch_size: int = 64

    # Data
    _X: List[List[Optional[Any]]]
    _Y: List[List[Optional[Any]]]
    _dataset = TrainValidationDataset

    _model: IClassifierModel

    def __init__(self,
                 preprocessor: IArticlePreprocessor,
                 articles: Articles,
                 max_article_length: int,
                 supported_themes: List[str],
                 theme_metrics: List[ThemeMetricAggregator],
                 model: IClassifierModel):
        super(LambdaCallback, self).__init__()

        self._preprocessor = preprocessor
        self._max_article_length = max_article_length
        self._themes = supported_themes

        self._prepare_data(articles)

        self._theme_metrics__ = theme_metrics
        self._model = model


    def train(self) -> TrainedModel:
        """
        Trained the given model with data passed in the constructor.
        """
        if self._trained:
            raise Exception("Multiple training is not supported.")

        theme_weights = ThemeWeights(self._theme_stats, self._theme_tokenizer)

        self.logger.info("Parameters:")
        self.logger.info("Batch size: %d", self.batch_size)

        self._model.train_model(theme_weights, self._dataset, self._article_tokenizer.voc_size, self)

        return TrainedModel(self._model, self._article_tokenizer, self._theme_tokenizer)


    def _prepare_data(self, articles: Articles):
        self._processed_articles = self._process_articles(articles)
        self._tokenize_articles()

        getLogger("Prepare the data.\n")

        self._dataset = TrainValidationDataset(
            self._X,
            self._Y,
            articles=self._processed_articles,
            validation_ratio=self.validation_ratio,
            batch_size=self.batch_size
        )

        self._data_analysis()


    def _process_articles(self, articles: Articles) -> Articles:
        self.logger.info("Preprocessing training articles.")
        return self._preprocessor.process_articles(articles)


    def _tokenize_articles(self):
        self.logger.info("Tokenizing training articles.")
        self._article_tokenizer = ArticleTextTokenizer(self._processed_articles, self._max_article_length)
        self._theme_tokenizer = ArticleThemeTokenizer(self._processed_articles)

        self._X = self._article_tokenizer.sequences
        self._Y = self._theme_tokenizer.one_hot_matrix


    def _data_analysis(self):
        self.logger.info("\nBasic Training Data Analysis")
        self.logger.info("-------------")

        self.logger.info("* Number of articles: %d", len(self._article_tokenizer.sequences))
        self.logger.info("* Size of vocabulary: %d", self._article_tokenizer.voc_size)

        for theme in self._themes:
            article_with_theme = self._processed_articles.articles_with_theme(theme).items
            stat = ThemeStat(theme, len(article_with_theme), self._processed_articles.count())
            self._theme_stats.append(stat)
            self.logger.info("'{}' {} / {} => Weights: (Positive: {}, : Negative: {})".format(theme, stat.article_of_theme_count, stat.total_article_count, stat.binary_weight_pos(), stat.binary_weight_neg()))

            article_train_count: int = self._dataset.articles_train.count()
            article_validation_count: int = self._dataset.articles_validation.count()
            articles_train_positive: int = self._dataset.articles_train.articles_with_theme(theme).count()
            articles_validation_positive: int = self._dataset.articles_validation.articles_with_theme(theme).count()

            self.logger.info("Validation ratio: %f", self.validation_ratio)
            self.logger.info("Train data: %d records (%d of theme = %f)", article_train_count, articles_train_positive,
                         articles_train_positive / article_train_count)
            self.logger.info("Validation data: %d records (%d of theme = %f)", article_validation_count,
                         articles_validation_positive, articles_validation_positive / article_validation_count)

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of an epoch by Keras.
        """
        # Custom metrics computed at the end of an epoch.
        predictor = ArticlePredictor(self.model,
                                     self._themes,
                                     self._preprocessor,
                                     self._article_tokenizer,
                                     self._theme_tokenizer)
        predictor.logger = logging.getLogger("PN-predictor-Trainer")
        predictor.logger.setLevel(logging.WARNING)

        predictions_validation: ArticlesPrediction = predictor.predict_preprocessed(self._dataset.articles_validation)
        predictions_train: ArticlesPrediction = predictor.predict_preprocessed(self._dataset.articles_train)

        for metric in self._theme_metrics__:
            metric.evaluate(predictions_train, predictions_validation, self._theme_tokenizer)

        for metric in self._theme_metrics__:
            metric.plot()

