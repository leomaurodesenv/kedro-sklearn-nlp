"""
This is a boilerplate pipeline 'training'
generated using Kedro 0.18.7
"""

import time
import logging
import pandas as pd
from typing import Dict, List, Tuple, Union

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV


LOGGER = logging.getLogger(__name__)


def _run_grid_search(model, params, train_X: any, train_y: any):
    # Testing
    search = GridSearchCV(model, params)
    search.fit(train_X, train_y)

    LOGGER.info(search.best_estimator_)
    LOGGER.info(search.best_estimator_.get_params()["classifier"])


def train_logistic_regression(train_X: any, train_y: any) -> None:
    # params = {
    #     "C": [10**-2, 10**-1, 10**0],
    #     "kernel": ["linear", "rbf", "sigmoid"],
    #     "class_weight": [None, {0:1,1:3}, {0:1,1:5}],
    # }

    # _run_grid_search(SVC(), params, train_X, train_y)
    pass


def train_random_forest(train_X: any, train_y: any) -> None:
    pass


def train_svc(train_X: any, train_y: any) -> SVC:
    """Train Support Vector Classifier (SVC)

    Args:
        train_X: train data
        train_y: train label
    Returns:
        model: sklearn.svm.SVC
    """
    runtime = time.time()
    model = SVC(kernel="rbf", gamma='auto')
    model.fit(train_X, train_y)
    runtime = time.time() - runtime

    # Logging
    LOGGER.info("## SVC Training")
    LOGGER.info("Runtime elapsed %.4f seconds" % runtime)

    return model
