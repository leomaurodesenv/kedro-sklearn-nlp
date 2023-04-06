"""
This is a boilerplate test file for pipeline 'preprocessing'
generated using Kedro 0.18.7.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""
from pathlib import Path

import pytest
from kedro.pipeline import Pipeline, node
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

from kedro.io import DataCatalog
from kedro.runner import SequentialRunner

from kedro_sklearn.pipelines.preprocessing.pipeline import create_pipeline


@pytest.fixture
def config_session():
    project_path = Path.cwd()
    bootstrap_project(project_path)
    return KedroSession.create(project_path=project_path, env="base")


class TestPreprocessingPipeline:
    """Preprocessing pipeline tests"""

    def test_pipeline_creation(self):
        """Test create_pipeline method"""
        pipeline = create_pipeline()
        assert isinstance(pipeline, Pipeline)

    def test_pipeline_run(self, config_session):
        """Test run the pipeline"""
        with config_session as session:
            session.run(pipeline_name="preprocessing")
        assert True

    def test_pipeline_run2(self, config_session):
        """Test run the pipeline"""
        # the sequential runner is the simplest. It runs one node at a time.
        runner = SequentialRunner()
        # pipeline = create_pipeline()
        pipeline = Pipeline(
            [
                node(lambda: range(100), None, "range"),
                node(lambda x: [i ** 2 for i in x], "range", "range**2"),
                node(lambda x: [i for i in x if i > 5000], "range**2", "range>5k"),
                node(lambda x: x[:5], "range>5k", "range>5k-head"),
                node(lambda x: sum(x) / len(x), "range>5k", "range>5k-mean"),
            ]
        )

        # to get up and running, you can use an empty catalog
        catalog = DataCatalog()

        runner.run(pipeline, catalog)
