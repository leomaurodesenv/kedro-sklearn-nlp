"""
This is a boilerplate pipeline 'preprocessing'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preprocess_train, preprocess_test


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_train,
                inputs=["train", "params:preprocessing.dataset"],
                outputs=["train_vectorizer", "train_X", "train_y"],
                name="preprocess_train_node",
            ),
            node(
                func=preprocess_test,
                inputs=["test", "train_vectorizer", "params:preprocessing.dataset"],
                outputs="test_X",
                name="preprocess_test_node",
            ),
        ]
    )
