"""
This is a boilerplate pipeline 'selection'
generated using Kedro 0.18.7
"""

import logging
import pandas as pd
from typing import Dict, List, Tuple, Union


LOGGER = logging.getLogger(__name__)


def model_selection(metric_name: str, **kargs) -> any:
    """Model selection

    Args:
        model_{index} (any): Model
        metric_{index} (str): Model metrics
        metric_name (str): Principal metric
    Returns:
        best_model (any): Best model according to the metric
    """
    idx = 0
    best_model = kargs["model_0"]
    best_score = kargs["metric_0"][metric_name]

    while True:
        if not f"model_{idx}" in kargs:
            break
        if kargs[f"metric_{idx}"][metric_name] > best_score:
            best_model = kargs[f"model_{idx}"]
            best_score = kargs[f"metric_{idx}"][metric_name]
        idx = int(idx + 1)

    # Logging
    LOGGER.info("## Best model")
    LOGGER.info(best_model)
    LOGGER.info(f"{metric_name}: {best_score:.4f}")

    return best_model


def model_prediction(model: any, X_test: any, test_set: pd.DataFrame):
    """Model prediction

    Args:
        model (any): Best model
        X_test (any): Input test data
        test_set (any): Test dataset
    Returns:
        submission (pd.DataFrame): Submission file
    """
    target = model.predict(X_test)
    id_test = test_set["id"].to_list()
    submission = {"id": id_test, "target": target}
    df_submission = pd.DataFrame(submission)

    return df_submission
