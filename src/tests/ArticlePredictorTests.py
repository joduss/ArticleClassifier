import unittest

from classifier.prediction.article_predictor import ArticlePredictor
from classifier.preprocessing.article_text_tokenizer import ArticleTextTokenizer
from classifier.preprocessing.article_theme_tokenizer import ArticleThemeTokenizer
from classifier.preprocessing.interface_article_preprocessor import IArticlePreprocessor
from data_models.article import Article
from data_models.articles import Articles
from tests.mock_model import MockModel


class ArticlesPredictorTests(unittest.TestCase):

    def test_articles_not_modified_by_predictor(self):
        """
        Test if articles fields 'themes' and 'verified_themes' are not modified
        by the predictor.
        :return:
        """

        tokenizer_init_article = Article(id="0",
                                         title="",
                                         summary="theme1 theme2 theme3",
                                         themes=["theme1", "theme2", "theme3"],
                                         verified_themes=["theme1", "theme2", "theme3"],
                                         predicted_themes=[])

        articleOne = Article(id="1",
                             title="",
                             summary="theme1 theme2",
                             themes=["one", "two"],
                             verified_themes=["one", "two", "three"],
                             predicted_themes=["three"])

        article_tokenizer = ArticleTextTokenizer(Articles([tokenizer_init_article]), 3)
        theme_tokenizer = ArticleThemeTokenizer(Articles([tokenizer_init_article]))


        predictor = ArticlePredictor(classifier_model=MockModel.get_model(),
                                     supported_themes=["theme1", "theme2", "theme3"],
                                     preprocessor=MockPreprocessor(),
                                     article_tokenizer=article_tokenizer,
                                     theme_tokenizer=theme_tokenizer)

        prediction = predictor.predict_preprocessed(Articles(article=articleOne))

        article_with_predictions = prediction.get_articles_with_predictions()[0]

        self.assertEqual(["one", "two"], article_with_predictions.themes)
        self.assertEqual(["one", "two", "three"], article_with_predictions.verified_themes)
        self.assertEqual(["theme1", "theme2"], article_with_predictions.predicted_themes)


class MockPreprocessor(IArticlePreprocessor):

    def process_articles(self, articles: Articles) -> Articles:
        return articles.deep_copy()


    def process_article(self, article: Article) -> Article:
        return article.copy()