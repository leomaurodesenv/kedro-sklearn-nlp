"""
This is a boilerplate pipeline 'selection'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import model_selection, model_prediction

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=model_selection,
                inputs=dict(
                    model_0="model_svc",
                    model_1="model_random_forest",
                    model_2="model_logistic_regression",
                    metric_0="metrics_svc",
                    metric_1="metrics_random_forest",
                    metric_2="metrics_logistic_regression",
                    metric_name=f"params:{metric}.metric",
                ),
                outputs=f"{metric}.selected_model",
                name=f"{metric}.model_selection_node",
            )
            for metric in ["f1", "accuracy"]
        ] + [
            node(
                func=model_prediction,
                inputs=dict(
                    model="f1.selected_model",
                    X_test="test_X",
                    test_set="test",
                ),
                outputs="submission",
                name="model_prediction_node",
            )
        ]
    )
