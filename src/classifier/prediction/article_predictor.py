from logging import getLogger
from typing import List

import tensorflow as tf
import tensorflow.keras

from classifier.prediction.ArticlesPrediction import ArticlesPrediction
from classifier.preprocessing.article_text_tokenizer import ArticleTextTokenizer
from classifier.preprocessing.article_theme_tokenizer import ArticleThemeTokenizer
from classifier.preprocessing.interface_article_preprocessor import IArticlePreprocessor
from data_models.article import Article
from data_models.articles import Articles


class ArticlePredictor:
    """
    Do theme predictions for articles.
    """

    CLASSIFIER_THRESHOLD: float = 0.5
    logger = getLogger()

    def __init__(self,
                 classifier_model: tf.keras.models.Model,
                 supported_themes: List[str],
                 preprocessor: IArticlePreprocessor,
                 article_tokenizer: ArticleTextTokenizer,
                 theme_tokenizer: ArticleThemeTokenizer):
        self.classifier_model: tf.keras.models.Model = classifier_model
        self.supported_themes: List[str] = supported_themes
        self.preprocessor: IArticlePreprocessor = preprocessor
        self.theme_tokenizer: ArticleThemeTokenizer = theme_tokenizer
        self.article_tokenizer: ArticleTextTokenizer = article_tokenizer


    def predict(self, articles_original: Articles) -> ArticlesPrediction:
        """
        Pre-processes articles, compute the predictions for each of them and aggregate the predictions into a
        ArticlesPrediction object, which is returned.
        :param articles_original: NON-preprocessed articles
        """
        predictions = ArticlesPrediction(self.theme_tokenizer, articles_original)
        processed_articles = Articles([article for article in self.preprocessor.process_articles(articles_original)])

        self.logger.debug("Will start predictions with keras model")
        matrix = self.article_tokenizer.transform_to_sequences(processed_articles)
        prediction_matrix = self.classifier_model.predict(matrix)
        self.logger.debug("Did predictions with keras model")

        idx = 0
        for prediction_vector in prediction_matrix:
            article_id = processed_articles[idx].id
            predictions.addPredictionsForArticle(prediction_vector, article_id)

            idx += 1

        # processed_articles = {article.id : article for article in self.preprocessor.process_articles(articles_original)}
        #
        # num_article_json = len(processed_articles)
        # num_processed_article = 0
        #
        # predictions = ArticlesPrediction(self.theme_tokenizer, articles_original)
        # article_orig: Article

        # for article_orig in articles_original:
        #     article_processed = processed_articles[article_orig.id]
        #     prediction = self.__predict_article(article_processed, self.supported_themes)
        #     predictions.add(article_orig, prediction)
        #     num_processed_article += 1
        #
        #     if num_processed_article % 10 == 0:
        #         print("Progress: " + str(num_processed_article) + "/" + str(num_article_json), end='\r')

        self.logger.info("Finished predicting themes for %d articles", articles_original.count())
        return predictions


    def predict_preprocessed(self, processed_articles: Articles) -> ArticlesPrediction:
        """
        Compute the predictions for articles of them and aggregate the predictions into a
        ArticlesPrediction object, which is returned.
        Articles must have been previously pre-processed!
        :param processed_articles: Preprocessed articles
        """
        predictions = ArticlesPrediction(self.theme_tokenizer, processed_articles)

        self.logger.debug("Will start predictions with keras model")
        matrix = self.article_tokenizer.transform_to_sequences(processed_articles)
        prediction_matrix = self.classifier_model.predict(matrix)
        self.logger.debug("Did predictions with keras model")

        idx = 0
        for prediction_vector in prediction_matrix:
            article_id = processed_articles[idx].id
            predictions.addPredictionsForArticle(prediction_vector, article_id)

            idx += 1

        self.logger.info("Finished predicting themes for %d articles", processed_articles.count())
        return predictions


    def __predict_article(self, article: Article, themes_to_classify: List[str]) -> List[float]:
        """
        Do prediction for a preprocessed article.
        :param article preprocessed article:
        :param themes_to_classify:
        :return:
        """
        vector = self.article_tokenizer.transform_to_sequence(article)
        return self.classifier_model.predict(vector)[0]
