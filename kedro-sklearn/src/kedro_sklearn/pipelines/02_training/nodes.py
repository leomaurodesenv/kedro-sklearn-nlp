"""
This is a boilerplate pipeline 'training'
generated using Kedro 0.18.7
"""

import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV, cross_validate


LOGGER = logging.getLogger(__name__)


def _run_grid_search(
    model: any, params: Dict[any, any], train_X: any, train_y: any, k: int = 5
) -> Tuple:
    """GridSearch Hyperparameter Tuning

    Args:
        model: Model
        params: Hyperparameters
        train_X: Train data
        train_y: Train label
        k: Number of validations
    Returns:
        metrics: Avg results
    """
    search = GridSearchCV(
        model, params, refit="f1_macro", cv=k, scoring=["accuracy", "f1_macro"]
    )
    search.fit(train_X, train_y)

    metrics = {
        "accuracy": np.mean(search.cv_results_["mean_test_accuracy"]),
        "f1": np.mean(search.cv_results_["mean_test_f1_macro"]),
    }

    # Logging
    LOGGER.info(search.best_estimator_)
    LOGGER.info(search.best_params_)

    return search.best_estimator_, metrics


def _run_k_fold(model: any, train_X: any, train_y: any, k: int = 5) -> Dict:
    """k-Fold Cross Validation

    Args:
        model: Model
        train_X: Train data
        train_y: Train label
        k: Number of validations
    Returns:
        metrics: Avg results
    """
    cv_results = cross_validate(
        model, train_X, train_y, cv=k, scoring=["accuracy", "f1_macro"]
    )
    metrics = {
        "accuracy": np.mean(cv_results["test_accuracy"]),
        "f1": np.mean(cv_results["test_f1_macro"]),
    }

    return metrics


def train_logistic_regression(train_X: any, train_y: any) -> Tuple:
    """Train Logistic Regression

    Args:
        train_X: Train data
        train_y: Train label
    Returns:
        model: sklearn.linear_model.LogisticRegression
        metrics: Dict of metrics
    """
    runtime = time.time()
    params = {
        "max_iter": [5_000],
        "penalty": [None, "l2"],
        "class_weight": [None, {0: 1, 1: 3}, {0: 1, 1: 5}],
    }
    model, metrics = _run_grid_search(
        model=LogisticRegression(), params=params, train_X=train_X, train_y=train_y
    )
    runtime = time.time() - runtime

    # Logging
    LOGGER.info("## Random Forest Training")
    LOGGER.info("Avg Accuracy %.4f seconds" % metrics["accuracy"])
    LOGGER.info("Avg F1-score %.4f seconds" % metrics["f1"])
    LOGGER.info("Runtime elapsed %.4f seconds" % runtime)

    return model, metrics


def train_random_forest(train_X: any, train_y: any) -> Tuple:
    """Train Random Forest

    Args:
        train_X: Train data
        train_y: Train label
    Returns:
        model: sklearn.ensemble.RandomForestClassifier
        metrics: Dict of metrics
    """
    runtime = time.time()
    model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=0)
    metrics = _run_k_fold(model=model, train_X=train_X, train_y=train_y)
    model.fit(train_X, train_y)
    runtime = time.time() - runtime

    # Logging
    LOGGER.info("## Random Forest Training")
    LOGGER.info("Avg Accuracy %.4f seconds" % metrics["accuracy"])
    LOGGER.info("Avg F1-score %.4f seconds" % metrics["f1"])
    LOGGER.info("Runtime elapsed %.4f seconds" % runtime)

    return model, metrics


def train_svc(train_X: any, train_y: any) -> Tuple:
    """Train Support Vector Classifier (SVC)

    Args:
        train_X: Train data
        train_y: Train label
    Returns:
        model: sklearn.svm.SVC
        metrics: Dict of metrics
    """
    runtime = time.time()
    model = SVC(kernel="rbf", gamma="auto", random_state=0)
    metrics = _run_k_fold(model=model, train_X=train_X, train_y=train_y)
    model.fit(train_X, train_y)
    runtime = time.time() - runtime

    # Logging
    LOGGER.info("## SVC Training")
    LOGGER.info("Avg Accuracy %.4f seconds" % metrics["accuracy"])
    LOGGER.info("Avg F1-score %.4f seconds" % metrics["f1"])
    LOGGER.info("Runtime elapsed %.4f seconds" % runtime)

    return model, metrics
