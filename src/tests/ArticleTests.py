import unittest

from data_models.article import Article
from data_models.articles import Articles


class ArticleTests(unittest.TestCase):

    def test_equal(self):
        article1 = Article("1", "title1", "sum1", ["th1"], ["th1", "th2"], [])
        article2 = Article("2", "title2", "sum1", ["th1"], ["th1", "th2"], [])
        article3 = Article("1", "title1", "sum1", ["th1"], ["th1", "th2"], ["extra_theme"])
        article4 = Article("1", "title1", "sum1", ["th1", "th1"], ["th1", "th2"], [])

        # self.assertNotEqual(article1, article2)
        self.assertNotEqual(article1, article3)
        self.assertNotEqual(article1, article4)

        article_eq1 = Article("3", "title1", "sum1", ["th1"], ["th1", "th2"], [])
        article_eq2 = Article("3", "title1", "sum1", ["th1"], ["th2", "th1"], [])
        article_eq3 = Article("3", "title1", "sum1", ["th1"], ["th2", "theme10", "th1"], [])
        article_eq4 = Article("3", "title1", "sum1", ["th1"], ["th1", "theme10", "th2"], [])

        self.assertEqual(article_eq1, article_eq2)
        self.assertEqual(article_eq3, article_eq4)

