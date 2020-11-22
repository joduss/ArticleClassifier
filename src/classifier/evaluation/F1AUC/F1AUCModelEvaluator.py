from dataclasses import dataclass
from typing import Dict, List

from classifier.evaluation.F1AUC.MetricF1AUC import MetricF1AUC
from classifier.evaluation.abstracts.ModelEvaluator import ModelEvaluator
from classifier.evaluation.metrics.AUC import AUC
from classifier.prediction.ArticlesPrediction import ArticlesPrediction
from classifier.preprocessing.article_theme_tokenizer import ArticleThemeTokenizer


@dataclass
class F1AUCModelEvaluator(ModelEvaluator):
    """
    Evaluates the classification per theme.
    """

    print_stats: bool

    def __init__(self, theme_tokenizer: ArticleThemeTokenizer = None, print_stats: bool = False):
        super().__init__(theme_tokenizer)
        self.print_stats = print_stats

    def evaluate(self, predictions: ArticlesPrediction, themes: List[str]) -> Dict[str, MetricF1AUC]:
        """
        Evaluate
        :param predictions:
        :param themes:
        """
        metrics = {}

        for theme in themes:
            metrics[theme] = self.evaluate_theme(theme, predictions)

        return metrics


    def evaluate_theme(self, theme: str, predictions: ArticlesPrediction) -> MetricF1AUC:
        """
        Print evaluation stats for the classification for a single theme.
        :param predictions: predictions object.
        :param theme: theme for which the evaluation must be performed
        """
        if self._theme_tokenizer_ is None:
            raise Exception("The theme tokenizer was not set.")

        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0

        for article in predictions.get_articles_with_predictions():
            if theme not in article.verified_themes:
                continue

            is_of_theme = theme in article.themes
            prediction = theme in article.predicted_themes

            if is_of_theme and prediction:
                true_positive += 1
            elif is_of_theme and not prediction:
                false_negative += 1
            elif not is_of_theme and prediction:
                false_positive += 1
            else:
                true_negative += 1

        if self.print_stats:
            print("\n")
            print("Evaluation of predictions for theme \"{}\"".format(theme))
            print("------")

            print("TP = {}".format(true_positive))
            print("TN = {}".format(true_negative))
            print("FP = {}".format(false_positive))
            print("FN = {}".format(false_negative))

        recall = 0
        precision = 0
        f_score = 0

        if (true_positive + false_negative) > 0:
            recall = true_positive / (true_positive + false_negative)

        if (true_positive + false_positive) > 0:
            precision = true_positive / (true_positive + false_positive)

        if (precision + recall) > 0:
            f_score = 2 * (precision * recall) / (precision + recall)

        auc: float = AUC.compute_auc(predictions, theme)

        if self.print_stats:
            print("Recall = {}".format(recall))
            print("Precision = {}".format(precision))
            print("F-scrore = {}".format(f_score))
            print("AUC = {}".format(auc))

        return MetricF1AUC(theme,
                           auc=auc,
                           f1=f_score,
                           precision=precision,
                           recall=recall)
