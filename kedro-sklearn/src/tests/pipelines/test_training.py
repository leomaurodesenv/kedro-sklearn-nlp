"""
This is a boilerplate test file for pipeline 'training'
generated using Kedro 0.18.7.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""
from pathlib import Path

import pytest
from kedro.pipeline import Pipeline
from sklearn.datasets import make_classification

from kedro_sklearn.pipelines.training import nodes, pipeline


class TestTrainingPipeline:
    """Training pipeline tests"""

    def test_pipeline_creation(self):
        """Test create_pipeline method"""
        pipeline_list = pipeline.create_pipeline()
        assert isinstance(pipeline_list, Pipeline)


@pytest.fixture
def train_set():
    """Get train dataset"""
    train_params = {"report_path": "data/08_reporting"}
    X, y = make_classification(
        n_samples=30,
        n_features=5,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
    )
    return X, y, train_params


class TestTrainingNodes:
    """Training nodes tests"""

    def test_train_svc(self, train_set):
        """Test train_svc method"""
        X, y, train_params = train_set
        model, metrics = nodes.train_svc(
            train_X=X, train_y=y, train_params=train_params
        )
        assert (model is not None) and isinstance(metrics, dict)

    def test_train_random_forest(self, train_set):
        """Test train_random_forest method"""
        X, y, train_params = train_set
        model, metrics = nodes.train_random_forest(
            train_X=X, train_y=y, train_params=train_params
        )
        assert (model is not None) and isinstance(metrics, dict)

    def test_train_logistic_regression(self, train_set):
        """Test train_logistic_regression method"""
        X, y, train_params = train_set
        model, metrics = nodes.train_logistic_regression(
            train_X=X, train_y=y, train_params=train_params
        )
        assert (model is not None) and isinstance(metrics, dict)
