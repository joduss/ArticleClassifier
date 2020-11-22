from dataclasses import dataclass


@dataclass
class ThemeStat:
    """
    Variables:
        theme: theme for which these statistics are
        article_count: number of articles of this theme
        total_article_count: total number of articles.
    """

    # Calculation of weight:
    # weight_for_0 = (1 / neg)*(total)/2.0
    # weight_for_1 = (1 / pos)*(total)/2.0
    #
    # Source: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#calculate_class_weights

    theme: str
    article_of_theme_count: int
    total_article_count : int

# TODO: https://towardsdatascience.com/text-classifier-with-multiple-outputs-and-multiple-losses-in-keras-4b7a527eb858

    def binary_weight_pos(self) -> float:
        """
        Returns the weight for the class "is of that theme".
        """
        return (1 / self.article_of_theme_count)*(self.total_article_count)/2.0

    def binary_weight_neg(self) -> float:
        """
        Returns the weight for the class "is NOT of that theme".
        """
        articles_of_not_theme_count = self.total_article_count - self.article_of_theme_count
        return (1 / articles_of_not_theme_count)*(self.total_article_count)/2.0


    # 3 classes:
    # A: 100 -> mean error = 10
    # B: 1000 -> mean error = 1
    # C: 500 -> mean error = 1
    #
    # total: 1600
    #
    # we want to have the mean = 12 / 3 = 4.