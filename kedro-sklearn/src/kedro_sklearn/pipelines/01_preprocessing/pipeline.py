"""
This is a boilerplate pipeline 'preprocessing'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocess_test, preprocess_train


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_train,
                inputs=dict(
                    train_set="train",
                    parameters="params:preprocessing.dataset",
                ),
                outputs=["feature_vectorizer", "train_X", "train_y"],
                name="preprocess_train_node",
            ),
            node(
                func=preprocess_test,
                inputs=dict(
                    test_set="test",
                    vectorizer="feature_vectorizer",
                    parameters="params:preprocessing.dataset",
                ),
                outputs="test_X",
                name="preprocess_test_node",
            ),
        ]
    )
