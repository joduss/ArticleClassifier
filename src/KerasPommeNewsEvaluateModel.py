from typing import List

from classifier.evaluation.F1AUC.F1AUCModelEvaluator import F1AUCModelEvaluator
from classifier.prediction.losses.weightedBinaryCrossEntropy import WeightedBinaryCrossEntropy
from data_models.articles import Articles
from tensorflow.keras.models import load_model

ARTICLE_JSON_FILE = "articles_{}.json"
LANG = "fr"
LANG_FULL = "french"
MODEL_PATH = "model.h5" # Relative path

LIMIT_ARTICLE_COUNT = None # None or a number.

SUPPORTED_THEMES: List[str] = ["computer", "smartphone"]


# Loads the articles
# ==================
articles_filepath = ARTICLE_JSON_FILE.format(LANG)

if (LIMIT_ARTICLE_COUNT is None):
    all_articles: Articles = Articles.from_file(articles_filepath)
else:
    all_articles: Articles = Articles.from_file(articles_filepath, LIMIT_ARTICLE_COUNT)


# Load the model
# ==================

model = load_model(MODEL_PATH, custom_objects={"WeightedBinaryCrossEntropy" : WeightedBinaryCrossEntropy()})

# Perform evaluation
# ==================

F1AUCModelEvaluator().evaluate(all_articles, SUPPORTED_THEMES)