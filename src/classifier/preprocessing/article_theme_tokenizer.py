from typing import List

import numpy as np
from keras_preprocessing.text import Tokenizer

from data_models.articles import Articles


class ArticleThemeTokenizer:

    '''
    List of themes in the same order as in the tokenizer, which corresponds as well
    as the index of theme in the prediction
    '''
    orderedThemes: List[str]
    themes_count: int
    tokenizer: Tokenizer


    def __init__(self, articles: Articles):
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(articles.themes())

        self.one_hot_matrix = self.tokenizer.texts_to_matrix(articles.themes())

        # Remove the first column, whose first col contains only 0s.
        self.one_hot_matrix = np.delete(arr=self.one_hot_matrix, obj=0, axis=1)

        # Create ordered list of theme as in tokenizer
        self.orderedThemes: List[str] = []

        for i in range(1, len(self.tokenizer.word_index) + 1):  # word_index start at 1, 0 is reserved.
            self.orderedThemes.append(self.tokenizer.index_word[i])

        self.themes_count = len(self.tokenizer.word_index)



    def index_of_theme(self, theme: str):
        return self.tokenizer.word_index[theme] - 1

    def theme_at_index(self, index: int):
        return self.tokenizer.index_word[index + 1]

    def boolean_vector_to_themes(self, prediction_vector: List[bool]) -> List[str]:

        themes: List[str] = []

        for idx in range(0, len(prediction_vector)):
            if prediction_vector[idx]:
                # +1 because the first index (0) is reserved by default.
                themes.append(self.tokenizer.index_word[idx + 1])

        return themes
