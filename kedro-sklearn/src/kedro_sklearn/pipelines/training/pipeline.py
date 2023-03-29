"""
This is a boilerplate pipeline 'training'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_svc


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_svc,
                inputs=["train_X", "train_y"],
                outputs="model_svc",
                name="train_svc_node",
            )
        ]
    )
