import os
from prefect import flow, task

from churn.main import preprocess, train, evaluate
from churn.packages.mlflow import mlflow_transition_model
from churn.packages.parameters import *

@task
def preprocess_data(min_date:str, max_date:str):
    return preprocess(min_date=min_date, max_date=max_date)

@task
def evaluate_model(min_date:str, max_date:str, stage:str):
    return evaluate(min_date=min_date, max_date=max_date, stage=stage)

@task
def train_new_model(min_date:str, max_date:str, split_ratio:float):
    return train(min_date=min_date, max_date=max_date, split_ratio=split_ratio)

@task
def mlflow_transition(current_stage:str, new_stage:str):
    return mlflow_transition_model(current_stage=current_stage, new_stage=new_stage)

@task
def notify():
    pass


@flow(name=PREFECT_FLOW_NAME)
def train_flow():
    pass
