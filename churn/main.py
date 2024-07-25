import numpy as np
import pandas as pd
from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse
from sklearn.model_selection import train_test_split
import datetime

from churn.packages.parameters import *
from churn.packages.mlflow import save_results, save_model, mlflow_, load_model
from churn.packages.model import initialise_model, evaluate_model
from churn.packages.data import clean_data, get_data_with_cache, load_data_to_bq, create_data
from churn.packages.preprocess import preprocess_features


def data_pipeline(date:datetime = CURRENT_DATE) -> None:
    """
    - Query the raw dataset for the oirginal data from BigQuery (2024-07)
    - Cache query results as a locl csv if it doesn't already exist locally
    - Create artificial data
    - Upload the data to APPEND to the original BigQuery data table
    """
    query = f"""
        SELECT {",".join(COLUMN_NAME_ROWS)}
        FROM `{PROJECT_ID}.{BQ_DATASET}.raw_data`
        WHERE Date = '2024-07'
    """

    data = create_data(query=query, cache_path=Path(LOCAL_PATH).joinpath(f"customers_06_2024.csv"))

    load_data_to_bq(data=data,
        gcp_project=PROJECT_ID,
        bq_dataset=BQ_DATASET,
        table='raw_data',
        truncate=False)

    return print("✅ data_pipeline done \n")



def preprocess(min_date:str= '2024-07', max_date:str= '2024-07') -> None:
    """
    - Query the raw dataset from the BigQuery Dataset
    - Cache query results as a locl csv if it doesn't already exist locally
    - Process query data
    - Store processed data on a different BigQuery table (truncate existing table data)
    - No need to cache processed data as csv (it will be cached when queries back from BigQuery during training)
    """
    min_date = parse(min_date).strftime('%Y-%m')
    max_date = parse(max_date).strftime('%Y-%m')

    query = f"""
        SELECT {",".join(COLUMN_NAME_ROWS)}
        FROM `{PROJECT_ID}.{BQ_DATASET}.raw_data`
        WHERE Date = '{min_date}'
        ORDER BY Date
    """

    #Fetch data
    data_query_cache_path = Path(LOCAL_PATH).joinpath(f"customers_{min_date}_{max_date}.csv")
    data_query = get_data_with_cache(
        query=query,
        gcp_project=PROJECT_ID,
        cache_path=data_query_cache_path,
        data_has_header=True
    )

    #Process data
    cleaned_data = clean_data(data_query)

    X = cleaned_data.drop(columns='Churn', axis=1)
    y = cleaned_data[['Churn']]

    X_processed = preprocess_features(X)

    processed_data = pd.DataFrame(np.concatenate((X_processed, y), axis=1))
    load_data_to_bq(
        data=processed_data,
        gcp_project=PROJECT_ID,
        bq_dataset=BQ_DATASET,
        table='processed',
        truncate=True
    )

    print("✅ preprocess() done \n")
    return None


@mlflow_
def train(
        min_date:str = '2024-07',
        max_date:str = '2024-07',
        split_ratio:float = 0.3,

    )-> float:
    """
    - Download the processed data from BQ(unless already cached locally)
    - Train the model on the data
    - Store the training results on MLflow
    - Return the accuracy of the model
    """

    print(Fore.MAGENTA + "\n⭐️ Use case: train" + Style.RESET_ALL)
    print(Fore.BLUE + "\nLoading preprocessed validation data..." + Style.RESET_ALL)

    query = f"""
        SELECT *
        FROM `{PROJECT_ID}.{BQ_DATASET}.processed`
        WHERE _25 = '{min_date}'
        ORDER BY _25
    """

    data_processed_cache_path = Path(LOCAL_PATH).joinpath(f"processed_{min_date}_{max_date}.csv")
    data_processed = get_data_with_cache(
        query=query,
        gcp_project=PROJECT_ID,
        cache_path=data_processed_cache_path,
        data_has_header=True
    )

    if data_processed.shape[0] < 10:
        print("❌ Not enough processed data retrieved to train on")
        return None

    X = data_processed.drop(columns=['_25','_27'])
    y = data_processed[['_27']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=42)

    model = initialise_model()
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)

    params = dict(
        context="train",
        date_start=min_date,
        date_end=max_date,
        row_count=len(X_train)
    )

    metrics = dict(
        key=score
    )

    save_results(params=params, metrics=metrics)

    save_model(model=model)

    print("✅ train() done \n")

    return score

@mlflow_
def evaluate(min_date:str ='2024-05', max_date:str ='2024-06', stage:str ='Production'):
    """
    Evaluate the current production model.
    """
    model = load_model(stage=stage)
    assert model is not None

    min_date = parse(min_date).strftime('%Y-%m')
    max_date = parse(max_date).strftime('%Y-%m')

    query = f"""
        SELECT * EXCEPT(_0)
        FROM `{PROJECT_ID}`.{BQ_DATASET}.processed`
        WHERE _25 BETWEEN '{min_date}' AND '{max_date}'
    """

    data_query_cache_path = Path(f"../local_data/customers_{min_date}_{max_date}.csv")
    data_query = get_data_with_cache(
        query=query,
        gcp_project=PROJECT_ID,
        cache_path=data_query_cache_path,
        data_has_header=True
    )

    X = data_query.drop(columns=['_27'])
    X = X.to_numpy()

    y = data_query['_27']
    y = y.to_numpy()

    metrics = evaluate_model(model=model, X=X, y=y)
    accuracy = metrics["accuracy"]

    params = dict(
            context="evaluate",
            training_set_size=data_query.shape[0],
            row_count=len(X)
        )

    save_results(params=params, metrics=metrics)

    print("✅ evaluation done.")

    return accuracy


def predicion(X_pred:pd.DataFrame) -> np.ndarray:
    """
    Make a prediction using the most recent model.
    """
    if X_pred is None:
        return None

    model = load_model()
    assert model is not None

    X_processed = preprocess_features(X_pred)
    y_pred = model.predict(X_processed)

    print("\n✅ prediction done: ", y_pred, y_pred.shape, "\n")

    return y_pred


if __name__ == '__main__':
    #preprocess(min_date='2024-07', max_date='2024-07')
    #data_pipeline()
    #train()
    pass
