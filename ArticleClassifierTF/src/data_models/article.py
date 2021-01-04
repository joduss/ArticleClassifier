from typing import List, Dict


class Article:

    __immutable: bool = False

    __id: str
    __title: str
    __summary: str

    __themes: List[str]
    __predicted_themes: List[str]
    __verified_themes: List[str]


    def __init__(self, id: str, title: str, summary: str, themes: List[str], predicted_themes: List[str],
                 verified_themes: List[str]):
        self.id = id
        self.title = title
        self.summary = summary
        self.themes = themes
        self.predicted_themes = predicted_themes
        self.verified_themes = verified_themes


    def copy(self):
        return Article(
            self.id,
            self.title,
            self.summary,
            self.themes.copy(),
            self.predicted_themes.copy(),
            self.verified_themes.copy(),
        )


    def title_and_summary(self) -> str:
        return self.title + ". " + self.summary


    def __eq__(self, other):
        if not isinstance(other, Article):
            return False

        other_article: Article = other

        self.themes.sort()
        self.verified_themes.sort()
        self.predicted_themes.sort()
        other_article.themes.sort()
        other_article.verified_themes.sort()
        other_article.predicted_themes.sort()

        return self.id == other_article.id \
               and self.summary == other_article.summary \
               and self.title == other_article.title \
               and self.themes == other_article.themes \
               and self.verified_themes == other_article.verified_themes \
               and self.predicted_themes == other_article.predicted_themes

    def make_immutable(self):
        self.__immutable = True


    @property
    def id(self):
        return self.__id

    @id.setter
    def id(self, id):
        self.__raise_if_immutable()
        self.__id = id

    @property
    def title(self):
        return self.__title

    @title.setter
    def title(self, value):
        self.__raise_if_immutable()
        self.__title = value

    @property
    def summary(self):
        return self.__summary

    @summary.setter
    def summary(self, value):
        self.__raise_if_immutable()
        self.__summary = value

    @property
    def themes(self):
        return self.__themes

    @themes.setter
    def themes(self, value):
        self.__raise_if_immutable()
        self.__themes = value

    @property
    def verified_themes(self):
        return self.__verified_themes

    @verified_themes.setter
    def verified_themes(self, value):
        self.__raise_if_immutable()
        self.__verified_themes = value


    def __raise_if_immutable(self):
        if self.__immutable:
            raise Exception("This article has been made immutable!")