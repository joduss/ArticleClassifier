import unittest

from classifier.preprocessing.article_theme_tokenizer import ArticleThemeTokenizer
from data_models.article import Article
from data_models.articles import Articles


class ArticleThemeTokenizerTests(unittest.TestCase):

    def test_boolean_vector_to_themes(self):
        article1 = Article("1", "title", "summary", ["theme1", "theme2", "theme3"], [], [])
        article2 = Article("2", "title", "summary", ["theme1", "theme4"], [], [])
        articles = Articles([article1, article2])

        tokenizer = ArticleThemeTokenizer(articles)

        self.assertEqual(4, tokenizer.themes_count)
        self.assertEqual(["theme1", "theme2", "theme3", "theme4"], tokenizer.orderedThemes)
        self.assertEqual(["theme1", "theme4"], tokenizer.boolean_vector_to_themes([True, False, False, True]))
        self.assertEqual([], tokenizer.boolean_vector_to_themes([False, False, False, False]))
        self.assertEqual(["theme3"], tokenizer.boolean_vector_to_themes([False, False, True, False]))