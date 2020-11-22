import unittest

from data_models.article import Article
from data_models.articles import Articles


class ArticlesTests(unittest.TestCase):

    @staticmethod
    def create_articles() -> Articles:
        article1 = Article(title="Title", summary="summary", themes=[], verified_themes=[], predicted_themes=[], id="1")
        article2 = Article(title="Title", summary="summary", themes=["T"], verified_themes=["T"], predicted_themes=[], id="2")
        article3 = Article(title="Title", summary="summary", themes=["T", "T2"], verified_themes=[], predicted_themes=[], id="3")
        article4 = Article(title="Title", summary="summary", themes=[], verified_themes=["T"], predicted_themes=[], id="4")
        article5 = Article(title="Title", summary="summary", themes=["T2"], verified_themes=["T"], predicted_themes=[], id="5")
        article6 = Article(title="Title", summary="summary", themes=["T", "T2", "T3"], verified_themes=["T", "T2", "T3"], predicted_themes=["T3"], id="6")

        return Articles([article1, article2, article3, article4, article5, article6])

    def test_articles_with_theme(self):
        articles = ArticlesTests.create_articles()

        filtered = articles.articles_with_theme("T2")

        self.assertEqual(3, filtered.count())
        self.assertTrue(articles.items[2] in filtered)
        self.assertTrue(articles.items[4] in filtered)
        self.assertTrue(articles.items[5] in filtered)
        self.assertFalse(articles.items[3] in filtered)

    def test_articles_with_all_verified_themes(self):
        articles = ArticlesTests.create_articles()
        filtered = articles.articles_with_all_verified_themes(["T", "T2"])

        self.assertEqual(1, filtered.count())
        self.assertTrue(articles.items[5] in filtered)
        self.assertFalse(articles.items[0] in filtered)

    def test_articles_with_any_verified_themes(self):
        articles = ArticlesTests.create_articles()
        filtered = articles.articles_with_any_verified_themes(["T", "T2", "T3"])

        self.assertEqual(4, filtered.count())
        self.assertFalse(articles[0] in filtered)
        self.assertTrue(articles[1] in filtered)
        self.assertFalse(articles[2] in filtered)
        self.assertTrue(articles[3] in filtered)
        self.assertTrue(articles[4] in filtered)
        self.assertTrue(articles[5] in filtered)

    def test_themes(self):
        themes = ArticlesTests.create_articles().themes()

        self.assertEqual(0, len(themes[0]))
        self.assertEqual(1, len(themes[1]))
        self.assertEqual(2, len(themes[2]))
        self.assertEqual(0, len(themes[3]))
        self.assertEqual(1, len(themes[4]))
        self.assertEqual(3, len(themes[5]))

        self.assertEqual("T", themes[5][0])
        self.assertEqual("T2", themes[5][1])
        self.assertEqual("T3", themes[5][2])

    def test_title_and_summary(self):
        articles = self.create_articles()

        self.assertEqual("Title. summary", articles.title_and_summary()[0])


    def test_deep_copy(self):
        articles = self.create_articles()
        articles_copy = articles.deep_copy()

        for i in range(0,articles.count()):
            article = articles.items[i]
            article_copy = articles_copy.items[i]

            self.assertEqual(article.id, article_copy.id)
            self.assertEqual(article.summary, article_copy.summary)
            self.assertEqual(article.title, article_copy.title)
            self.assertEqual(article.themes, article_copy.themes)
            self.assertEqual(article.verified_themes, article_copy.verified_themes)
            self.assertEqual(article.predicted_themes, article_copy.predicted_themes)

            article_copy.predicted_themes.append("T4")
            self.assertNotEqual(article.predicted_themes, article_copy.predicted_themes)

            article.predicted_themes.append("T4")
            self.assertEqual(article.predicted_themes, article_copy.predicted_themes)

    def test_substraction(self):
        articles = self.create_articles()

        articles_to_remove = Articles(self.create_articles()[0:2])

        filtered_articles = articles - articles_to_remove

        self.assertEqual(filtered_articles.count() + 2, articles.count())
        self.assertFalse(filtered_articles.contains(articles_to_remove[0].id))
        self.assertFalse(filtered_articles.contains(articles_to_remove[1].id))
        self.assertTrue(filtered_articles.contains(articles[2].id))
        self.assertTrue(filtered_articles.contains(articles[3].id))
        self.assertTrue(filtered_articles.contains(articles[4].id))
        self.assertTrue(filtered_articles.contains(articles[5].id))


if __name__ == '__main__':
    unittest.main()
