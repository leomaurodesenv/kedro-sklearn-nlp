"""
This is a boilerplate pipeline 'preprocessing'
generated using Kedro 0.18.7
"""

import logging
import pandas as pd
from typing import Dict, List, Tuple, Union
from sklearn.feature_extraction.text import CountVectorizer


LOGGER = logging.getLogger(__name__)


def preprocess_train(train_set: pd.DataFrame) -> Tuple[any, any]:
    """Preprocesses

    Args:
        train_set: Raw data.
    Returns:
        vectorizer: sklearn.feature_extraction.text.CountVectorizer
        X: numpy array of the corpus
    """
    # Vectorizer the dataset
    corpus = train_set["text"]
    vectorizer = CountVectorizer(lowercase=True, ngram_range=(1, 2), max_features=10_000)
    X = vectorizer.fit_transform(corpus)

    # Logging
    LOGGER.info("## Train preprocessing")
    LOGGER.info("corpus size: %d" % train_set.shape[0])
    LOGGER.info("vector features: %d" % X.shape[1])

    return vectorizer, X


def preprocess_test(test_set: pd.DataFrame, vectorizer: any) -> any:
    """Preprocesses test

    Args:
        test_set: Raw data.
    Returns:
        X: numpy array of the corpus
    """
    # Vectorizer the dataset
    corpus = test_set["text"]
    X = vectorizer.transform(corpus)

    # Logging
    LOGGER.info("## Test preprocessing")
    LOGGER.info("corpus size: %d" % test_set.shape[0])
    LOGGER.info("vector features: %d" % X.shape[1])

    return X
