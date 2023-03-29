"""
This is a boilerplate pipeline 'training'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_svc, train_random_forest, train_logistic_regression


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_svc,
                inputs=dict(
                    train_X="train_X",
                    train_y="train_y",
                ),
                outputs="model_svc",
                name="train_svc_node",
            ),
            node(
                func=train_random_forest,
                inputs=dict(
                    train_X="train_X",
                    train_y="train_y",
                ),
                outputs="model_random_forest",
                name="train_random_forest_node",
            ),
            node(
                func=train_logistic_regression,
                inputs=dict(
                    train_X="train_X",
                    train_y="train_y",
                ),
                outputs="model_logistic_regression",
                name="train_logistic_regression_node",
            ),
        ]
    )
