from classifier.models.IClassifierModel import IClassifierModel
from classifier.preprocessing.article_text_tokenizer import ArticleTextTokenizer
from classifier.preprocessing.article_theme_tokenizer import ArticleThemeTokenizer


class TrainedModel:

    theme_tokenizer: ArticleThemeTokenizer
    article_tokenizer: ArticleTextTokenizer
    model: IClassifierModel

    def __init__(self, model: IClassifierModel,
                  article_tokenizer : ArticleTextTokenizer,
                  theme_tokenizer: ArticleThemeTokenizer):
        self.theme_tokenizer = theme_tokenizer
        self.article_tokenizer = article_tokenizer
        self.model = model

    def save(self, directory: str):
        self.model.save_model(directory)
        self.article_tokenizer.save(f"{directory}{self.model.get_model_name()}-article-tokenizer.json")
        self.theme_tokenizer.save(f"{directory}{self.model.get_model_name()}-theme_tokenizer.json")


    def load(self, directory: str):
        self.model.load_model(directory)