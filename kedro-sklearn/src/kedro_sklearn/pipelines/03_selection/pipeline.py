"""
This is a boilerplate pipeline 'selection'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import model_prediction, model_selection


def create_pipeline(**kwargs) -> Pipeline:
    pipeline_instance = pipeline(
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
                    metric_name=f"params:metric",
                ),
                outputs=f"selected_model",
                name=f"model_selection_node",
            ),
            node(
                func=model_prediction,
                inputs=dict(
                    model="selected_model",
                    X_test="test_X",
                    test_set="test",
                ),
                outputs="submission",
                name="model_prediction_node",
            ),
        ]
    )
    pipeline_f1 = pipeline(
        pipe=pipeline_instance,
        inputs=dict(
            model_svc="model_svc",
            model_random_forest="model_random_forest",
            model_logistic_regression="model_logistic_regression",
            metrics_svc="metrics_svc",
            metrics_random_forest="metrics_random_forest",
            metrics_logistic_regression="metrics_logistic_regression",
            test_X="test_X",
            test="test",
        ),
        namespace="f1",
    )
    pipeline_accuracy = pipeline(
        pipe=pipeline_instance,
        inputs=dict(
            model_svc="model_svc",
            model_random_forest="model_random_forest",
            model_logistic_regression="model_logistic_regression",
            metrics_svc="metrics_svc",
            metrics_random_forest="metrics_random_forest",
            metrics_logistic_regression="metrics_logistic_regression",
            test_X="test_X",
            test="test",
        ),
        namespace="accuracy",
    )
    return pipeline_f1 + pipeline_accuracy
