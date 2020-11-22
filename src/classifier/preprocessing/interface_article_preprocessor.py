from typing import List

from data_models.article import Article
from data_models.articles import Articles


class IArticlePreprocessor:

    def process_articles(self, articles: Articles) -> Articles:
        """
        Remove stopwords and do lemmatization on each article, for the specified language.
        :param articles: data_models to process
        :param LANG: language of the data_models
        :return: processed data_models
        """
        raise Exception("Not implemented")

    def process_article(self, article: Article) -> Article:
        """
        Remove stopwords and do lemmatization on one single article, for the specified language.
        :param article: article to process
        :return: processed article
        """
        raise Exception("Not implemented")