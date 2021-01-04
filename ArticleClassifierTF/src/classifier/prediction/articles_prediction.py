from typing import Dict, List

from classifier.preprocessing.article_theme_tokenizer import ArticleThemeTokenizer
from data_models.article import Article
from data_models.articles import Articles


class ArticlesPrediction:

    raw_predictions: Dict[str, List[float]] = {}
    __theme_tokenizer: ArticleThemeTokenizer
    articles: Articles = Articles()

    def __init__(self, theme_tokenizer: ArticleThemeTokenizer, articles: Articles):
        self.__theme_tokenizer = theme_tokenizer
        self.articles = articles

    def addPredictionsForArticle(self, predictions: List[float], article_id: str):
        """
        Add the predictions for that article.
        :param article_id:
        :param predictions:
        """
        self.raw_predictions[article_id] = predictions

    def get_articles_with_predictions(self, threshold: float = 0.5) -> Articles:
        return self.__apply_on_articles(threshold)


    def __apply_on_article(self, article: Article, threshold: float):
        """
        Apply the predictions on an article.
        :param article:
        :param threshold: Min probability to consider a theme as positively predicted.
        """
        if article.id not in self.raw_predictions.keys():
            raise Exception("No prediction found for that article (%s)", article.id)

        article.predicted_themes = self.__transform_to_themes(self.raw_predictions[article.id], threshold)

    def __apply_on_articles(self, threshold: float) -> Articles:
        """
        Apply the predictions on articles.
        :param threshold: Min probability to consider a theme as positively predicted.
        """
        articles = self.articles.deep_copy()

        for article in articles:
            self.__apply_on_article(article, threshold)

        return articles


    def __transform_to_themes(self, predictions: List[float], threshold: float) -> List[str]:
        """
        Transforms predictions that are under a form of probabilities into a list of them in a string form.
        :param predictions: Predictions for a single article
        :param threshold: Min probability to consider a theme as positively predicted.
        :return:
        """
        boolean_vector = list(map(lambda probability: probability >= threshold, predictions))
        return self.__theme_tokenizer.boolean_vector_to_themes(boolean_vector)

