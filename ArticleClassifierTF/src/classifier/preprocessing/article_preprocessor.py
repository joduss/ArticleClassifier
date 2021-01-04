from typing import List

import spacy
from pathos.multiprocessing import ProcessingPool as Pool

from classifier.preprocessing.interface_article_preprocessor import IArticlePreprocessor
from data_models.article import Article
from data_models.articles import Articles

spacier = spacy.load("fr_core_news_sm")

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')


class ArticlePreprocessor(IArticlePreprocessor):
    """
    Preprocessor cleaning the article / data_models
    """

    language: str = ''

    def __init__(self, language: str):
        self.language = language
        self.stop_words = set(stopwords.words(self.language))

    def process_articles(self, articles: Articles) -> Articles:
        """
        Remove stopwords and do lemmatization on each article, for the specified language.
        :param articles: data_models to process
        :param LANG: language of the data_models
        :return: processed data_models
        """
        p = Pool(8)
        return Articles(p.map(self.process_article, articles.items))


    def process_article(self, article: Article) -> Article:
        """
        Remove stopwords and do lemmatization on one single article, for the specified language.
        :param article: article to process
        :return: processed article
        """
        article_copy = article.copy()

        article_copy.title = self.__process_text(article.title).lower()
        article_copy.summary = self.__process_text(article.summary).lower()

        return article_copy

    def __process_text(self, text: str) -> str:
        """
        Remove stopwords and do lemmatization on a text, for the specified language
        :param text: article to process
        :param LANG: language of the article
        :return: processed article
        """

        text_tokenized = nltk.tokenize.word_tokenize(text, self.language)
        text_tokenized = [w for w in text_tokenized if not w in self.stop_words]
        text_processed = " ".join(text_tokenized)

        # lemmatizing the data_models return a list of lemmas.
        lemmas = spacier(text_processed)
        lemmas = [lemma.lemma_ for lemma in lemmas if lemma.lemma_ not in ["'", "'", "’"]]

        return " ".join(lemmas)
