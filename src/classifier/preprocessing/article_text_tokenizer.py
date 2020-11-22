from typing import Any, List, Optional

from keras_preprocessing.text import Tokenizer
from tensorflow import keras

from data_models.article import Article
from data_models.articles import Articles


class ArticleTextTokenizer:

    voc_size: int
    document_count: int
    sequences: List[List[Optional[Any]]]

    def __init__(self, articles: Articles, max_article_length: int):
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(articles.title_and_summary())
        self.max_article_length: int = max_article_length

        self.sequences = self.transform_to_sequences(articles)
        self.voc_size = len(self.tokenizer.word_index) + 1  # +1 because we pad with 0.
        self.document_count = self.tokenizer.document_count

    def transform_to_sequences(self, preprocessed_articles: Articles) -> List[List[Optional[Any]]]:
        """Transform articles content to a padded vector of length "max_article_length"."""
        matrix = self.tokenizer.texts_to_sequences(preprocessed_articles.title_and_summary())
        matrix = keras.preprocessing.sequence.pad_sequences(matrix,
                                                            value=0,
                                                            padding='post',
                                                            maxlen=self.max_article_length)
        return matrix

    def transform_to_sequence(self, preprocessed_article: Article):
        """Transform a article content to a padded vector of length "max_article_length"."""
        vector = self.tokenizer.texts_to_sequences([preprocessed_article.title_and_summary()])
        vector = keras.preprocessing.sequence.pad_sequences(vector,
                                                            value=0,
                                                            padding='post',
                                                            maxlen=self.max_article_length)
        return vector