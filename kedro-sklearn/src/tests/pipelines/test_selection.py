"""
This is a boilerplate test file for pipeline 'selection'
generated using Kedro 0.18.7.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""
import pandas as pd
from kedro.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from kedro_sklearn.pipelines.selection import nodes, pipeline

from .test_training import train_set


class TestSelectionPipeline:
    """Selection pipeline tests"""

    def test_pipeline_creation(self):
        """Test create_pipeline method"""
        pipeline_list = pipeline.create_pipeline()
        assert isinstance(pipeline_list, Pipeline)


class TestSelectionNodes:
    """Selection nodes tests"""

    def test_model_selection(self):
        """Test model_selection method"""
        model_1 = "model_1"
        selected_model = nodes.model_selection(
            metric_name="acc",
            model_0="model_0",
            model_1=model_1,
            metric_0={"acc": 0.5},
            metric_1={"acc": 0.9},
        )
        assert selected_model == model_1

    def test_model_prediction(self, train_set):
        """Test model_prediction method"""
        X_test, y, _ = train_set
        model = LogisticRegression()
        model.fit(X_test, y)
        test_set = pd.DataFrame(data={"id": y})
        # generating the submission file
        submission = nodes.model_prediction(
            model=model, X_test=X_test, test_set=test_set
        )
        assert isinstance(submission, pd.DataFrame) and (
            submission.shape[0] == y.shape[0]
        )
