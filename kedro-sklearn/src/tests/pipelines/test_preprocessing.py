"""
This is a boilerplate test file for pipeline 'preprocessing'
generated using Kedro 0.18.7.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""
from pathlib import Path

import pandas as pd
import pytest
from kedro.config import ConfigLoader
from kedro.framework.context import KedroContext
from kedro.framework.hooks import _create_hook_manager
from kedro.framework.project import settings
from kedro.pipeline import Pipeline
from kedro.runner import SequentialRunner

from kedro_sklearn.pipelines.preprocessing import nodes, pipeline


@pytest.fixture
def config_loader():
    """Get config loader"""
    return ConfigLoader(conf_source=str(Path.cwd() / settings.CONF_SOURCE))


@pytest.fixture
def project_context(config_loader):
    """Get kedro context"""
    return KedroContext(
        package_name="kedro_sklearn",
        project_path=Path.cwd(),
        config_loader=config_loader,
        hook_manager=_create_hook_manager(),
    )


class TestPreprocessingPipeline:
    """Preprocessing pipeline tests"""

    def test_pipeline_creation(self):
        """Test create_pipeline method"""
        pipeline_list = pipeline.create_pipeline()
        assert isinstance(pipeline_list, Pipeline)

    def test_pipeline_run(self, project_context):
        """Test run the pipeline"""
        runner = SequentialRunner()
        pipeline_list = pipeline.create_pipeline()
        runner.run(pipeline_list, project_context.catalog)


@pytest.fixture
def dataset():
    """Get train dataset"""
    dataset = pd.DataFrame(
        data={
            "text": ["happy test", "sad test", "really sad test"],
            "target": [1, 0, 0],
        }
    )
    parameters = {
        "text_column": "text",
        "target_column": "target",
    }
    return dataset, parameters


class TestPreprocessingNodes:
    """Preprocessing nodes tests"""

    def test_preprocess_train_vectorizer(self, dataset):
        """Test preprocess_train method"""
        train_set, parameters = dataset
        vectorizer, _, _ = nodes.preprocess_train(
            train_set=train_set, parameters=parameters
        )
        assert vectorizer is not None

    def test_preprocess_train_arrays(self, dataset):
        """Test preprocess_train method"""
        train_set, parameters = dataset
        _, X, y = nodes.preprocess_train(train_set=train_set, parameters=parameters)
        assert X.shape[0] == y.shape[0]

    def test_preprocess_test(self, dataset):
        """Test preprocess_test method"""
        train_set, parameters = dataset
        vectorizer, X, _ = nodes.preprocess_train(
            train_set=train_set, parameters=parameters
        )
        test_X = nodes.preprocess_test(
            test_set=train_set, vectorizer=vectorizer, parameters=parameters
        )
        assert (X.shape[0] == test_X.shape[0]) and (X.shape[1] == test_X.shape[1])
