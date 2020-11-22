import logging
from logging import getLogger
from typing import Any, List, Optional

from tensorflow.python.keras.callbacks import LambdaCallback

from classifier.Data.TrainValidationDataset import TrainValidationDataset
from classifier.evaluation.abstracts.ThemeMetricAggregator import ThemeMetricAggregator
from classifier.prediction.ArticlesPrediction import ArticlesPrediction
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
    __supported_themes: List[str]

    __preprocessor: IArticlePreprocessor
    __processed_articles: Articles
    __max_article_length: int

    __article_tokenizer__: ArticleTextTokenizer
    __theme_tokenizer__: ArticleThemeTokenizer

    __theme_stats: List[ThemeStat] = []

    __theme_metrics__: List[ThemeMetricAggregator]

    __trained: bool = False

    validation_ratio: float = 0.2
    batch_size: int = 64

    # Data
    __X__: List[List[Optional[Any]]]
    __Y__: List[List[Optional[Any]]]
    __dataset__ = TrainValidationDataset

    __model__: IClassifierModel

    def __init__(self,
                 preprocessor: IArticlePreprocessor,
                 articles: Articles,
                 max_article_length: int,
                 supported_themes: List[str],
                 theme_metrics: List[ThemeMetricAggregator],
                 model: IClassifierModel):
        super(LambdaCallback, self).__init__()

        self.__preprocessor = preprocessor
        self.__max_article_length = max_article_length
        self.__supported_themes = supported_themes

        self.__prepare_data(articles)

        self.__theme_metrics__ = theme_metrics
        self.__model__ = model


    def train(self) -> TrainedModel:
        """
        Trained the given model with data passed in the constructor.
        :param model:
        """
        getLogger("\n\n---------------------\n")
        getLogger("Prepare to train the model.\n")
        if self.__trained:
            raise Exception("Multiple training is not supported.")

        getLogger().info("Data input:")
        getLogger().info("* Number of articles: %d", len(self.__article_tokenizer.sequences))
        getLogger().info("* Size of vocabulary: %d", self.__article_tokenizer.voc_size)
        getLogger().info("* Number of themes: %d", self.__theme_tokenizer.themes_count)
        getLogger().info("\n")

        theme_weights = ThemeWeights(self.__theme_stats, self.__theme_tokenizer)

        self.__dataset__ = TrainValidationDataset(
            self.__X__,
            self.__Y__,
            articles=self.__processed_articles,
            validation_ratio=self.validation_ratio,
            batch_size=self.batch_size
        )

        getLogger().info("Parameters:")
        getLogger().info("Batch size: %d", self.batch_size)
        getLogger().info("validation ratio: %d", self.validation_ratio)

        self.__model__.train_model(theme_weights, self.__dataset__, self.__article_tokenizer.voc_size, self)

        return TrainedModel(self.__model__, self.__article_tokenizer, self.__theme_tokenizer)


    def __prepare_data(self, articles: Articles):
        self.__processed_articles = self.__process_articles(articles)
        self.__tokenize_articles()
        self.__data_analysis()


    def __process_articles(self, articles: Articles) -> Articles:
        getLogger().info("Preprocessing training articles.")
        return self.__preprocessor.process_articles(articles)


    def __tokenize_articles(self):
        getLogger().info("Tokenizing training articles.")
        self.__article_tokenizer = ArticleTextTokenizer(self.__processed_articles, self.__max_article_length)
        self.__theme_tokenizer = ArticleThemeTokenizer(self.__processed_articles)

        self.__X__ = self.__article_tokenizer.sequences
        self.__Y__ = self.__theme_tokenizer.one_hot_matrix


    def __data_analysis(self):
        getLogger().info("\nBasic Data Analysis")
        getLogger().info("-------------")

        for theme in self.__supported_themes:
            article_with_theme = self.__processed_articles.articles_with_theme(theme).items
            stat = ThemeStat(theme, len(article_with_theme), self.__processed_articles.count())
            self.__theme_stats.append(stat)
            getLogger().info("'{}' {} / {} => Weights: (Positive: {}, : Negative: {})".format(theme, stat.article_of_theme_count, stat.total_article_count, stat.binary_weight_pos(), stat.binary_weight_neg()))


    def on_epoch_end(self, epoch, logs=None):

        # Custom metrics computed at the end of an epoch.
        predictor = ArticlePredictor(self.model,
                                     self.__supported_themes,
                                     self.__preprocessor,
                                     self.__article_tokenizer,
                                     self.__theme_tokenizer)
        predictor.logger = getLogger("pouf")
        predictor.logger.setLevel(logging.CRITICAL)

        predictions_validation: ArticlesPrediction = predictor.predict_preprocessed(self.__dataset__.articles_validation)
        predictions_train: ArticlesPrediction = predictor.predict_preprocessed(self.__dataset__.articles_train)

        for metric in self.__theme_metrics__:
            metric.evaluate(predictions_train, predictions_validation, self.__theme_tokenizer)


        for metric in self.__theme_metrics__:
            metric.plot()

