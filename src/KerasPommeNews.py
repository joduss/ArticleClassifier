from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from logging import getLogger
from typing import List

from classifier.evaluation.F1AUC.F1AUCModelEvaluator import F1AUCModelEvaluator
from classifier.evaluation.F1AUC.ThemeMetricF1AUCAggregator import ThemeMetricF1AUCAggregator
from classifier.models.ClassifierModel4 import ClassifierModel4
from classifier.models.ClassifierModel5 import ClassifierModel5
from classifier.prediction.article_predictor import ArticlePredictor
from classifier.models.ClassifierModel2 import ClassifierModel2
from classifier.models.IClassifierModel import IClassifierModel
from classifier.preprocessing.article_preprocessor_swift import ArticlePreprocessorSwift
from classifier.training.TrainedModel import TrainedModel
from classifier.training.Trainer import Trainer
from data_models.articles import Articles

print("\n\n\n####################################\n####################################")

############################################
# Configuration
############################################

# DATA CONFIGURATION
# ------------------

ARTICLE_JSON_FILE = "input/articles_{}.json"
LANG = "fr"
LANG_FULL = "french"

OUTPUT_DIR = "output/"

# supportedThemes: List[str] = ["android", "ios", "windows", "macos", "otherOS"]
# SUPPORTED_THEMES: List[str] = ["smartphone", "computer", "tablet"]
SUPPORTED_THEMES: List[str] = ["iphone"]

# MACHINE LEARNING CONFIGURATION
# ------------------------------

PREPROCESSOR = ArticlePreprocessorSwift()
DATASET_BATCH_SIZE = 128
ARTICLE_MAX_WORD_COUNT = 150
TRAIN_RATIO = 0.7
VALIDATION_RATIO = 0.1  # TEST is 1 - TRAIN_RATIO - VALIDATION_RATIO
VOCABULARY_MAX_SIZE = 50000  # not used for now!

# BEHAVIOUR CONFIGURATION
LIMIT_ARTICLES_TRAINING = False  # True or False
LIMIT_ARTICLES_PREDICTION = None  # None or a number

DO_COMPARISONS = False

############################################
# App initialization
############################################
logging.basicConfig(level=logging.DEBUG,
                    format='%(levelname)-8s %(module)-10s:  %(message)s',
                    datefmt='%m-%d %H:%M')
debugLogger = getLogger()
debugLogger.info("info")
debugLogger.warning("warning")
debugLogger.debug("debug")

TEST_RATIO = 1 - VALIDATION_RATIO - TRAIN_RATIO

############################################
# Data loading
############################################


# Loading the file
# ============================
debugLogger.info("Loading the file")

articles_filepath = ARTICLE_JSON_FILE.format(LANG)

if LIMIT_ARTICLES_TRAINING:
    all_articles: Articles = Articles.from_file(articles_filepath, 600)
else:
    all_articles: Articles = Articles.from_file(articles_filepath)

all_articles.shuffle()

for article in all_articles:
    article.make_immutable()


# Data filtering and partitionning
# ============================

articles_train: Articles = all_articles.articles_with_all_verified_themes(SUPPORTED_THEMES).deep_copy()

# Removal of all unsupported themes and keep only data_models who have at least one supported theme.
# -----------
debugLogger.info("Filtering and spliting data for testing and training.")

for article in articles_train.items:
    article.themes = [value for value in article.themes if value in SUPPORTED_THEMES]
    article.verified_themes = [value for value in article.verified_themes if value in SUPPORTED_THEMES]
    article.predicted_themes = [value for value in article.predicted_themes if value in SUPPORTED_THEMES]

debugLogger.info(
    "Removed %d data_models over %d without any supported themes. Left %d",
    all_articles.count() - articles_train.count(),
    all_articles.count(),
    articles_train.count()
)

# Split the article between training and test (train -> training + validation)
articles_train = articles_train.subset_ratio(TRAIN_RATIO)
articles_test = (all_articles - articles_train).articles_with_any_verified_themes(SUPPORTED_THEMES)

debugLogger.info("Train data: %d records", articles_train.count())
debugLogger.info("Test data: %d records", articles_test.count())


################################################################################################
# Data Analysis Section
################################################################################################

# For more advanced data analysis.

################################################################################################
# Machine Learning Section
################################################################################################

debugLogger.info("\n\nStarting Machine Learning")
debugLogger.info("-------------------------")

# Creation/settings of a model
# ============================

# For brands
# model = modelCreator.create_model(embedding_output_dim=128, intermediate_dim=256, last_dim=64, epochs=70)

# For device type
# model = modelCreator.create_model(embedding_output_dim=128, intermediate_dim=256, last_dim=256, epochs=20)
# model = modelCreator.create_model(embedding_output_dim=128, intermediate_dim=256, last_dim=256, epochs=20)



# do_ theme_weight for each theme!
theme_metric = ThemeMetricF1AUCAggregator(themes=SUPPORTED_THEMES,
                                          evaluator=F1AUCModelEvaluator())


#classifierModel: IClassifierModel = ClassifierModel3()
classifierModel: IClassifierModel = ClassifierModel5(OUTPUT_DIR)

trainer: Trainer = Trainer(preprocessor=PREPROCESSOR,
                           articles=articles_train,
                           max_article_length=ARTICLE_MAX_WORD_COUNT,
                           supported_themes=SUPPORTED_THEMES,
                           theme_metrics=[theme_metric],
                           model=classifierModel)
trainer.batch_size = DATASET_BATCH_SIZE
trainer.validation_ratio = VALIDATION_RATIO

trained_model = trainer.train()
trained_model.save(OUTPUT_DIR)



################################################################################################
# Classify unclassified data_models
################################################################################################

predictor = ArticlePredictor(trained_model.model.get_keras_model(),
                             SUPPORTED_THEMES,
                             PREPROCESSOR,
                             trained_model.article_tokenizer,
                             trained_model.theme_tokenizer)


# Evaluation of the model with test dataset
# ============================

debugLogger.info("Evaluation of the model with the test dataset.")
test_predictions = predictor.predict(articles_test)

evaluator = F1AUCModelEvaluator(trained_model.theme_tokenizer, print_stats=True)

evaluator.evaluate(
    test_predictions,
    SUPPORTED_THEMES
)

# Prediction for all articles
# ============================

articles_to_predict = all_articles

if LIMIT_ARTICLES_PREDICTION is not None:
    debugLogger.info("Limiting the number of articles to predict to " + str(LIMIT_ARTICLES_PREDICTION))
    articles_to_predict = all_articles.subset(LIMIT_ARTICLES_PREDICTION)


all_articles_predicted = predictor.predict(articles_to_predict).get_articles_with_predictions()
if LIMIT_ARTICLES_PREDICTION is not None:
    all_articles_predicted.save(f"{OUTPUT_DIR}predictions_limit_{LIMIT_ARTICLES_PREDICTION}.json")
else:
    all_articles_predicted.save(f"{OUTPUT_DIR}predictions.json")

debugLogger.info("End of program.")


theme_metric.plot(True)
k=input("press close to exit")