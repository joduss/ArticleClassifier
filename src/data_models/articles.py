from __future__ import annotations

import random
import math
import time
from typing import Callable, Dict, List, TextIO

import json as jsonModule

from data_models.article import Article
from data_models.transformation.article_transformer import ArticleTransformer
from utilities.utility import intersection


class MetaArticles(type):

    @property
    def items(cls) -> List[Article]:
        return cls.items


class Articles(object, metaclass=MetaArticles):

    jsonObject: Dict
    items: List[Article]

    def __init__(self, articles: List[Article] = None, article: Article = None):
        if articles is not None and isinstance(articles, list):
            self.items = articles
        elif article is not None and isinstance(article, Article):
            self.items = [article]
        elif article is None and articles is None:
            pass
        else:
            raise Exception("article or articles must be provided. NOT BOTH either!")

    def __iter__(self):
        return self.items.__iter__()

    def add(self, article: Article):
        self.items.append(article)

    @staticmethod
    def from_file(path: str, limit: int = None) -> Articles:
        with open(path, "r", encoding="utf-8") as file:
            try:
                return Articles.load_articles(file, limit)
            except:
                time.sleep(5)
                return Articles.load_articles(file, limit)



    @staticmethod
    def load_articles(file: TextIO, limit: int = None) -> Articles:
        """
            Creates an object data_models from a json file containing data_models.
        :param file:
        :param limit:
        """
        json = jsonModule.loads(file.read())

        if limit is not None:
            json = json[0:limit]

        articles: List[Article] = []

        for jsonArticle in json:
            articles.append(ArticleTransformer.transform_to_article(jsonArticle))

        return Articles(articles)


    def save(self, filepath: str):
        with open(filepath, 'w', encoding="utf-8") as outfile:
            jsonModule.dump([ArticleTransformer.transform_to_json(article) for article in self.items], outfile, indent=4, ensure_ascii=False)

    # def inherit_predictions(self, articles: Articles):
    #     original_dic = { i.id : i for i in self.items }
    #
    #     for predicted_article in articles:
    #         original_article: Article = original_dic[predicted_article.id]
    #         original_article.predicted_themes = predicted_article.predicted_themes
    #
    #     print("done")

    def subset(self, size: int or None) -> Articles:
        """
        Creates a subset of the articles.
        :param size: An integer or None.
        :return: The subset of size 'size' or all articles if size is None..
        """
        if size is None:
            return self
        return Articles(self.items[0:size])


    def subset_ratio(self, ratio: float) -> Articles:
        return self.subset(size=math.ceil(self.count() * ratio))


    def articles_with_theme(self, theme: str) -> Articles:
        """
        Returns articles which have the given theme in the list of themes (property 'themes')
        :param theme: theme that must be present
        :return: articles having the given theme.
        """
        return Articles(
            list(
                filter(lambda article: theme in article.themes, self.items)
            )
        )

    def get_by_id(self, article_id: str):
        for article in self.items:
            if article.id is article_id:
                return article

        raise Exception("Not found.")

    # def filter(self, filter_function: Callable[[Article], bool]) -> Articles:
    #     return Articles(
    #             list(
    #                 filter(
    #                     lambda article: filter_function(article),
    #                     self.items,
    #                 )
    #             )
    #         )

    def articles_with_all_verified_themes(self, themes: List[str]) -> Articles:
        """
        Returns a new Articles instance containing articles whose verified themes are containing a given list of themes.
        :param themes: All themes that must be have been verified in the articles.
        :return:
        """
        return Articles(
            list(
                filter(
                    lambda article: (len(intersection(themes, article.verified_themes)) == len(themes)),
                    self.items
                )
            )
        )


    def articles_with_any_verified_themes(self, themes: List[str]) -> Articles:
        """
        Returns a new Articles instance containing articles whose verified themes are containing at least one of a
        theme from a given list of themes.
        :param themes: Themes. At least one must be present in the article verified themes.
        :return:
        """
        return Articles(
            list(
                filter(
                    lambda article: (len(intersection(themes, article.verified_themes)) > 0),
                    self.items
                )
            )
        )


    def themes(self) -> List[str]:
        """
        Returns the list of themes for each articles according to the order of articles.
        """
        return list(
            map(lambda article: article.themes, self.items)
        )


    def title_and_summary(self) -> List[str]:
        return list(
            map(lambda article: article.title_and_summary(), self.items)
        )


    def count(self) -> int:
        return len(self.items)


    def copyEachArticle(self) -> List[Article]:
        return list(
            map(lambda article: article.copy(), self.items)
        )


    def deep_copy(self) -> Articles:
        return Articles(self.copyEachArticle())


    def shuffle(self):
        random.shuffle(self.items)


    def contains(self, article_id: str) -> bool:
        for item in self.items:
            if item.id == article_id:
                return True
        return False


    def __sub__(self, other):
        if not isinstance(other, Articles):
            raise Exception("Must be type Articles")

        # just for typing
        other_articles: Articles = other

        filtered = [article for article in self.items if not other_articles.contains(article.id)]
        return Articles(filtered)


    def __getitem__(self, idx):
        return self.items[idx]