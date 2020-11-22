from typing import Dict, List

from classifier.preprocessing.article_theme_tokenizer import ArticleThemeTokenizer
from data_models.ThemeStat import ThemeStat


class ThemeWeights:

    theme_stats: List[ThemeStat]
    theme_tokenizer: ArticleThemeTokenizer

    def __init__(self, theme_stats: List[ThemeStat], theme_tokenizer: ArticleThemeTokenizer):
        self.theme_stats = theme_stats
        self.theme_tokenizer = theme_tokenizer


    def weight_list(self) -> List[float]:
        """
        Returns a list of weight for each theme, ordered by theme index.
        """
        theme_weight: List[float] = list([])

        #raise Exception("To review")

        for theme in self.theme_tokenizer.orderedThemes:
            stat = [stat for stat in self.theme_stats if stat.theme == theme][0]
            theme_weight.append(stat.binary_weight_pos())

        return theme_weight


    def weights_of_theme(self, theme_idx: int) -> Dict[int, float]:
        """
        Returns the weights for a theme under the form {0 : VAL_1, 1 : VAL_2}
        :param theme_idx: index of the theme
        """
        theme = self.theme_tokenizer.theme_at_index(theme_idx)
        theme_stat = list(filter(lambda stat: stat.theme == theme, self.theme_stats))

        if len(theme_stat) == 0:
            raise Exception("Theme {} not found.".format(theme))

        if len(theme_stat) > 1:
            raise Exception("Theme {} found multiple times.".format(theme))

        return {0 : theme_stat[0].binary_weight_neg(),
                1 : theme_stat[0].binary_weight_pos()}


    def weight_array(self) -> List[List[float]]:
        theme_weight_array: List[List[float]] = []

        # raise Exception("To review")

        for theme in self.theme_tokenizer.orderedThemes:
            stat = [stat for stat in self.theme_stats if stat.theme == theme][0]

            theme_weight = [0,0]
            theme_weight[0] = stat.binary_weight_neg()
            theme_weight[1] = stat.binary_weight_pos()

            theme_weight_array.append(theme_weight)

        return theme_weight_array