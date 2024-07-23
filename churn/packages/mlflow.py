import time
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression
from churn.packages.parameters import *


def save_results(params: dict, metrics: dict) -> None:
    """
    - Stores the parameters and metrics of the model to MLflow
    """
    if params is not None:
        mlflow.log_params(params)
    if metrics is not None:
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

    print("✅ Results saved on mlflow")

    return None


def save_model(model):
    """
    - Stores the model to MLflow
    """
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path='model',
        registered_model_name=MODEL_NAME
    )

    print("✅ Model saved on mlflow")

    return None


def mlflow_transition_model(current_stage:str, new_stage:str):
    '''
    Transitions the new model from current_stage to new_stage and archives the old model into the
    '''

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = MlflowClient()

    version = client.get_latest_versions(name=MODEL_NAME, stages=[current_stage])

    if not version:
        print(f"\n❌ No model found with name {MODEL_NAME} in stage {current_stage}")
        return None

    client.transition_model_version_stage(
        name=MODEL_NAME, version=version,
        stage=new_stage,
        archive_existing_versions=True)

    return print(f"✅ Model {MODEL_NAME} (version {version[0].version}) transitioned from {current_stage} to {new_stage}")


def mlflow_(func):
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
        return results

    return wrapper


def load_model(stage='Production') -> LogisticRegression:
    """
    Load the 'Production' model from MLflow
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = MlflowClient()

    try:
        version = client.get_latest_versions(name=MODEL_NAME, stages=[stage])
        model_uri = version[0].source
        assert model_uri is not None
    except:
        print(f"No model named {MODEL_NAME} found in stage {stage}")
        return None

    model = mlflow.tensorflow.load_model(model_uri=model_uri)
    print(f"✅ Model {MODEL_NAME} successfully loaded")

    return model
