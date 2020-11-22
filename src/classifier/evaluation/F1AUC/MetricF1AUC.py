

class MetricF1AUC:

    theme: str
    f1: float
    auc: float
    precision: float
    recall: float

    def __init__(self, theme: str, auc: float, f1: float, precision: float, recall: float):
        self.theme = theme
        self.auc = auc
        self.f1 = f1
        self.precision = precision
        self.recall = recall
