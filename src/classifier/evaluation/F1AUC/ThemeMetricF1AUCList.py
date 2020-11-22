from typing import List, Dict

from classifier.evaluation.F1AUC.MetricF1AUC import MetricF1AUC


class ThemeMetricF1AUCList:

    metrics: Dict[str, List[MetricF1AUC]]

    def __init__(self):
        self.metrics = {}

    def add(self, theme: str, metric: MetricF1AUC):
        if theme not in self.metrics:
            self.metrics[theme] = []

        self.metrics[theme].append(metric)

    def get_f1(self, theme: str) -> List[float]:
        if theme not in self.metrics:
            raise Exception("Theme no found" + theme)

        return list(map(lambda metric: metric.f1, self.metrics[theme]))


    def get_precision(self, theme: str) -> List[float]:
        if theme not in self.metrics:
            raise Exception("Theme no found" + theme)

        return list(map(lambda metric: metric.precision, self.metrics[theme]))


    def get_recall(self, theme: str) -> List[float]:
        if theme not in self.metrics:
            raise Exception("Theme no found" + theme)

        return list(map(lambda metric: metric.recall, self.metrics[theme]))


    def get_auc(self, theme: str) -> List[float]:
        if theme not in self.metrics:
            raise Exception("Theme no found" + theme)

        return list(map(lambda metric: metric.auc, self.metrics[theme]))
