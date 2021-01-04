import unittest
from typing import List

from classifier.preprocessing.article_preprocessor_swift import ArticlePreprocessorSwift
from classifier.preprocessing.interface_article_preprocessor import IArticlePreprocessor
from data_models.article import Article
from data_models.articles import Articles


class ArticlesPreprocessorTests(unittest.TestCase):

    processors: List[IArticlePreprocessor] = [ArticlePreprocessorSwift()]

    def test_all_processors(self):

        for processor in self.processors:
            self.__test_single_processor(processor)

    def __test_single_processor(self, processor: IArticlePreprocessor):
        articles = self.__create_articles()
        articles_copied = self.__create_articles()

        articles_processed = processor.process_articles(articles)

        # Checking that the articles were not modified.
        for article in range(0, articles.count()):
            self.assertEqual(articles[0], articles_copied[0])
            self.assertEqual(articles[1], articles_copied[1])

        article_processed_ids = [article.id for article in articles_processed]
        article_processed_ids.sort()

        article_ids = [article.id for article in articles]
        article_ids.sort()

        self.assertEqual(article_ids, article_processed_ids)



    @staticmethod
    def __create_articles() -> Articles:
        article1 = Article("1", "Il s'agit d'un titre", "Le résumé numéro un.", ["theme1"], ["theme1", "old_prediction"], [])
        article2 = Article("2", "Ce sont deux titres", "Le résumé numéro deux.", ["theme1", "theme2", "theme3"], ["other_theme"], [])

        return Articles([article1, article2])

