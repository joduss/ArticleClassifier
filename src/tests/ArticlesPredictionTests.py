import unittest

from classifier.prediction.ArticlesPrediction import ArticlesPrediction
from classifier.preprocessing.article_theme_tokenizer import ArticleThemeTokenizer
from data_models.article import Article
from data_models.articles import Articles


class ArticlesPredictionTests(unittest.TestCase):

    def testApplyOnArticlesDefaultThreshold(self):
        article1 = Article("1", "title", "summary", ["theme1"], ["theme1", "old_prediction"], ["theme1"])
        article2 = Article("2", "title", "summary", ["theme1", "theme2"], ["other__old_predicted_theme"], ["theme1", "theme2", "theme3"])

        # article 3 is not used for test, but is necessary for the tokenizer to know the theme3.
        article3 = Article("3", "title", "summary", ["theme3"], [], [])

        articles = Articles([article1, article2])

        theme_tokenizer = ArticleThemeTokenizer(Articles([article1, article2, article3]))

        predictions = ArticlesPrediction(theme_tokenizer, articles)
        predictions.addPredictionsForArticle([0.1, 0.7, 0], article1.id)
        predictions.addPredictionsForArticle([0.4, 0.89, 0.99], article2.id)


        # Apply prediction with standard threshold
        predicted_articles = predictions.get_articles_with_predictions()
        predicted_articles_one = predicted_articles[0]
        predicted_articles_two = predicted_articles[1]

        self.assertEqual(1, len(predicted_articles_one.predicted_themes))
        self.assertFalse("theme1" in predicted_articles_one.predicted_themes)
        self.assertTrue("theme2" in predicted_articles_one.predicted_themes)
        self.assertFalse("theme3" in predicted_articles_one.predicted_themes)

        self.assertEqual(2, len(predicted_articles_two.predicted_themes))
        self.assertFalse("theme1" in predicted_articles_two.predicted_themes)
        self.assertTrue("theme2" in predicted_articles_two.predicted_themes)
        self.assertTrue("theme3" in predicted_articles_two.predicted_themes)


        # Check that the 'verified themes' and 'themes' are not touched!
        self.assertEqual(1, len(predicted_articles_one.themes))
        self.assertTrue("theme1" in predicted_articles_one.themes)
        self.assertFalse("theme2" in predicted_articles_one.themes)
        self.assertFalse("theme3" in predicted_articles_one.themes)

        self.assertEqual(2, len(predicted_articles_two.themes))
        self.assertTrue("theme1" in predicted_articles_two.themes)
        self.assertTrue("theme2" in predicted_articles_two.themes)
        self.assertFalse("theme3" in predicted_articles_two.themes)

        self.assertEqual(1, len(predicted_articles_one.verified_themes))
        self.assertTrue("theme1" in predicted_articles_one.verified_themes)
        self.assertFalse("theme2" in predicted_articles_one.verified_themes)
        self.assertFalse("theme3" in predicted_articles_one.verified_themes)

        self.assertEqual(3, len(predicted_articles_two.verified_themes))
        self.assertTrue("theme1" in predicted_articles_two.verified_themes)
        self.assertTrue("theme2" in predicted_articles_two.verified_themes)
        self.assertTrue("theme3" in predicted_articles_two.verified_themes)


        # Apply prediction with custom threshold
        predicted_articles = predictions.get_articles_with_predictions(0.09)

        predicted_articles_one = predicted_articles[0]
        predicted_articles_two = predicted_articles[1]

        self.assertEqual(2, len(predicted_articles_one.predicted_themes))
        self.assertTrue("theme1" in predicted_articles_one.predicted_themes)
        self.assertTrue("theme2" in predicted_articles_one.predicted_themes)
        self.assertFalse("theme3" in predicted_articles_one.predicted_themes)

        self.assertEqual(3, len(predicted_articles_two.predicted_themes))
        self.assertTrue("theme1" in predicted_articles_two.predicted_themes)
        self.assertTrue("theme2" in predicted_articles_two.predicted_themes)
        self.assertTrue("theme3" in predicted_articles_two.predicted_themes)