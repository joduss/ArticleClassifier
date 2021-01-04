from classifier.preprocessing.article_theme_tokenizer import ArticleThemeTokenizer


class ModelEvaluator:

    _theme_tokenizer_: ArticleThemeTokenizer

    def __init__(self, theme_tokenizer: ArticleThemeTokenizer = None):
        self._theme_tokenizer_ = theme_tokenizer

    def set_theme_tokenizer(self, theme_tokenizer: ArticleThemeTokenizer):
        self._theme_tokenizer_ = theme_tokenizer
