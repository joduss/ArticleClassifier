import logging
from typing import List

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import Plot
from classifier.evaluation.F1AUC.F1AUCModelEvaluator import F1AUCModelEvaluator
from classifier.evaluation.F1AUC.ThemeMetricF1AUCList import ThemeMetricF1AUCList
from classifier.evaluation.abstracts.ThemeMetricAggregator import ThemeMetricAggregator
from classifier.evaluation.abstracts.ModelEvaluator import ModelEvaluator
from classifier.prediction.ArticlesPrediction import ArticlesPrediction
from classifier.preprocessing.article_theme_tokenizer import ArticleThemeTokenizer


class ThemeMetricF1AUCAggregator(ThemeMetricAggregator):

    perThemeMetricsValidation: ThemeMetricF1AUCList = ThemeMetricF1AUCList()
    perThemeMetricsTrain: ThemeMetricF1AUCList = ThemeMetricF1AUCList()

    def __init__(self, themes: List[str], evaluator: F1AUCModelEvaluator):
        super(ThemeMetricAggregator, self).__init__()
        self.__supported_themes__ = themes
        self.evaluator = evaluator
        logging.getLogger('matplotlib').setLevel(logging.WARNING)


    def evaluate(self,
                 predictions_train: ArticlesPrediction,
                 predictions_validation: ArticlesPrediction,
                 theme_tokenizer: ArticleThemeTokenizer):
        self.evaluator.set_theme_tokenizer(theme_tokenizer)

        for theme in self.__supported_themes__:
            metric_val = self.evaluator.evaluate_theme(theme, predictions_validation)
            self.perThemeMetricsValidation.add(theme, metric_val)

            metric_train = self.evaluator.evaluate_theme(theme, predictions_train)
            self.perThemeMetricsTrain.add(theme, metric_train)


    def plot(self, block: bool = False):
        fig: Figure = plt.figure("plot_score_per_theme", figsize=[15,10])

        plt.rcParams['axes.grid'] = True
        plt.clf()
        plt.ion()


        i = 0
        for theme in self.__supported_themes__:
            f1_validation = self.perThemeMetricsValidation.get_f1(theme)
            f1_train = self.perThemeMetricsTrain.get_f1(theme)

            auc_validation = self.perThemeMetricsValidation.get_auc(theme)
            auc_train = self.perThemeMetricsTrain.get_auc(theme)

            precision_validation = self.perThemeMetricsValidation.get_precision(theme)
            precision_train = self.perThemeMetricsTrain.get_precision(theme)

            recall_validation = self.perThemeMetricsValidation.get_recall(theme)
            recall_train = self.perThemeMetricsTrain.get_recall(theme)

            ax = fig.subplots(2, 2, sharex="all", sharey="all")
            x = range(0, len(f1_train))

            if len(self.__supported_themes__) == 1:
                # Single theme
                ax[0, 0].plot(x, f1_validation, label="f1-validation")
                ax[0, 0].plot(x, f1_train, label="f1-train")

                ax[0, 1].plot(x, auc_validation, label="auc-validation")
                ax[0, 1].plot(x, auc_train, label="auc-train")

                ax[1, 0].plot(x, precision_validation, label="precision-validation")
                ax[1, 0].plot(x, precision_train, label="precision-train")

                ax[1, 1].plot(x, recall_validation, label="recall-validation")
                ax[1, 1].plot(x, recall_train, label="recall-train")

                ax[0, 0].set_title("F1 - " + theme)
                ax[0, 1].set_title("AUC")
                ax[1, 0].set_title("precision")
                ax[1, 1].set_title("recall")

                ax[0, 1].legend()
                ax[0, 0].legend()
                ax[1, 1].legend()
                ax[1, 0].legend()
            else:
                ax = fig.subplots(len(self.__supported_themes__), 2, sharex="all", sharey="all")

                # Multiple themes
                ax[i, 0].plot(x, f1_validation, label="f1-validation")
                ax[i, 0].plot(x, f1_train, label="f1-train")

                ax[i, 1].plot(x, auc_validation, label="auc-validation")
                ax[i, 1].plot(x, auc_train, label="auc-train")

                ax[i, 0].set_title(theme)
                ax[i, 1].set_title(theme)
                ax[i, 1].legend()
                ax[i, 0].legend()
        
            i+=1

        fig.autofmt_xdate()
        plt.show(block=block)
        plt.pause(0.05)

        #fig.show(block=False)
