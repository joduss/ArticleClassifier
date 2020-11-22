from sklearn import metrics

from classifier.prediction.ArticlesPrediction import ArticlesPrediction


class AUC:

    @staticmethod
    def compute_auc(predictions: ArticlesPrediction, theme: str):

        y_true = []
        y_pred = []

        for article in predictions.articles:
            if theme in article.themes:
                y_true.append(1)
            else:
                y_true.append(0)

            y_pred.append(predictions.raw_predictions[article.id][0])

        return metrics.roc_auc_score(y_true=y_true, y_score=y_pred)
