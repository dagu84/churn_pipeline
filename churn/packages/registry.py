import time
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from packages.params import *

def save_results(params: dict, metrics: dict) -> None:
    """
    - Stores the parameters and metrics of the model to MLflow
    """
    if params is not None:
        mlflow.log_params(params)
    if metrics is not None:
        mlflow.log_metric(metrics)

    print("✅ Results saved on mlflow")

    return None


def save_model(model, signature, input):
    """
    - Stores the model to MLflow
    """
    mlflow.sklearn.log_model(
        model=model,
        artifact_path='model',
        signature=signature,
        input_example=input,
        registered_model_name=MODEL_NAME
    )

    print("✅ Model saved on mlflow")

    return None


def mflow_transition_model(current_stage='None', new_staging='Staging'):
    pass


def mlflow_run(func):
    """
    Generic function to log params and results of to MLflow long with the model
    Args:
        - func (function): Function you want to run within the MLflow run
        - params (dict, optional): Params to add to the run in MLflow. Defaults to None.
        - context (str, optional): Param describing the context of the run. Defaults to "Train".
    """

    def wrapper(*args, **kwargs):
        mlflow.end_run()
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name=MFLOW_EXPERIEMENT)

        with mlflow.start_run():
            mlflow.sklearn.autolog()
            results = func(*args, **kwargs)

        print("✅ mlflow_run auto-log done")
