"""
This is a boilerplate pipeline 'training'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import train_logistic_regression, train_random_forest, train_svc


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_svc,
                inputs=dict(
                    train_X="train_X",
                    train_y="train_y",
                    train_params="params:train",
                ),
                outputs=["model_svc", "metrics_svc"],
                name="train_svc_node",
            ),
            node(
                func=train_random_forest,
                inputs=dict(
                    train_X="train_X",
                    train_y="train_y",
                    train_params="params:train",
                ),
                outputs=["model_random_forest", "metrics_random_forest"],
                name="train_random_forest_node",
            ),
            node(
                func=train_logistic_regression,
                inputs=dict(
                    train_X="train_X",
                    train_y="train_y",
                    train_params="params:train",
                ),
                outputs=["model_logistic_regression", "metrics_logistic_regression"],
                name="train_logistic_regression_node",
            ),
        ]
    )
