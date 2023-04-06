"""
This is a boilerplate pipeline 'training'
generated using Kedro 0.18.7
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from kedro.extras.datasets.matplotlib import MatplotlibWriter
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.svm import SVC

LOGGER = logging.getLogger(__name__)


def _classification_report(
    model_name: str,
    report_path: str,
    true_y: any,
    pred_y: any,
    labels: any = None,
    target_names: any = None,
) -> None:
    """Classification report image generator

    Args:
        model_name (str): Model name
        report_path (str): Report path
        true_y (list): True labels
        train_y (list): Predicted labels
        labels (list): List of records
        target_names (list): Names of each label
    Returns:
        None
    """
    report = classification_report(
        true_y, pred_y, labels=labels, target_names=target_names, output_dict=True
    )

    report_plot = sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True)
    report_fig = report_plot.get_figure()
    report_writer = MatplotlibWriter(
        filepath=f"{report_path}/classification_report-{model_name}.png"
    )
    report_writer.save(report_fig)


def _run_grid_search(
    model: any, params: Dict[any, any], train_X: any, train_y: any, k: int = 5
) -> Tuple:
    """GridSearch Hyperparameter Tuning

    Args:
        model (any): Model
        params (dict): Hyperparameters
        train_X (any): Train data
        train_y (any): Train label
        k (int): Number of validations
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
        model (any): Model
        train_X (any): Train data
        train_y (any): Train label
        k (int): Number of validations
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


def train_logistic_regression(train_X: any, train_y: any, train_params: Dict) -> Tuple:
    """Train Logistic Regression

    Args:
        train_X (any): Train data
        train_y (any): Train label
        train_params (dict): Training parameters
    Returns:
        model (sklearn.linear_model.LogisticRegression)
        metrics (dict): Dict of metrics
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

    # Reporting
    pred_y = model.predict(train_X)
    _classification_report(
        model_name="logistic_regression",
        report_path=train_params["report_path"],
        true_y=train_y,
        pred_y=pred_y,
    )

    # Logging
    LOGGER.info("## Random Forest Training")
    LOGGER.info("Avg Accuracy %.4f seconds" % metrics["accuracy"])
    LOGGER.info("Avg F1-score %.4f seconds" % metrics["f1"])
    LOGGER.info("Runtime elapsed %.4f seconds" % runtime)

    return model, metrics


def train_random_forest(train_X: any, train_y: any, train_params: Dict) -> Tuple:
    """Train Random Forest

    Args:
        train_X (any): Train data
        train_y (any): Train label
        train_params (dict): Training parameters
    Returns:
        model (sklearn.ensemble.RandomForestClassifier)
        metrics (dict): Dict of metrics
    """
    runtime = time.time()
    model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=0)
    metrics = _run_k_fold(model=model, train_X=train_X, train_y=train_y)
    model.fit(train_X, train_y)
    runtime = time.time() - runtime

    # Reporting
    pred_y = model.predict(train_X)
    _classification_report(
        model_name="random_forest",
        report_path=train_params["report_path"],
        true_y=train_y,
        pred_y=pred_y,
    )

    # Logging
    LOGGER.info("## Random Forest Training")
    LOGGER.info("Avg Accuracy %.4f seconds" % metrics["accuracy"])
    LOGGER.info("Avg F1-score %.4f seconds" % metrics["f1"])
    LOGGER.info("Runtime elapsed %.4f seconds" % runtime)

    return model, metrics


def train_svc(train_X: any, train_y: any, train_params: Dict) -> Tuple:
    """Train Support Vector Classifier (SVC)

    Args:
        train_X (any): Train data
        train_y (any): Train label
        train_params (dict): Training parameters
    Returns:
        model (sklearn.svm.SVC)
        metrics (dict): Dict of metrics
    """
    runtime = time.time()
    model = SVC(kernel="rbf", gamma="auto", random_state=0)
    metrics = _run_k_fold(model=model, train_X=train_X, train_y=train_y)
    model.fit(train_X, train_y)
    runtime = time.time() - runtime

    # Reporting
    pred_y = model.predict(train_X)
    _classification_report(
        model_name="svc",
        report_path=train_params["report_path"],
        true_y=train_y,
        pred_y=pred_y,
    )

    # Logging
    LOGGER.info("## SVC Training")
    LOGGER.info("Avg Accuracy %.4f seconds" % metrics["accuracy"])
    LOGGER.info("Avg F1-score %.4f seconds" % metrics["f1"])
    LOGGER.info("Runtime elapsed %.4f seconds" % runtime)

    return model, metrics
