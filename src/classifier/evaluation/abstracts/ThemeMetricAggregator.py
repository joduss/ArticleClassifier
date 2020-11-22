from classifier.evaluation.abstracts.ModelEvaluator import ModelEvaluator
from classifier.prediction.ArticlesPrediction import ArticlesPrediction
from classifier.preprocessing.article_theme_tokenizer import ArticleThemeTokenizer
from data_models.articles import Articles


class ThemeMetricAggregator:

    def plot(self):
        raise Exception("Not implemented")

    def evaluate(self,
                 predictions_train: ArticlesPrediction,
                 predictions_validation: ArticlesPrediction,
                 theme_tokenizer: ArticleThemeTokenizer):
        raise Exception("Not implemented")
