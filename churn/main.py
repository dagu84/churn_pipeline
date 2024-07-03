import numpy as np
import pandas as pd
from pathlib import Path
from dateutil.parser import parse

from churn.packages.params import *
from churn.packages.data import clean_data
from churn.packages.preprocess import preprocess_features

def preprocess(min_date:str= '2024-05', max_date:str= '2024-06') -> None:
    """
    - Query the raw dataset from the BigQuery Dataset
    - Cache query results as a locl csv if it doesn't already exist locally
    - Process query data
    - Store processed data on a different BigQuery table (truncate existing table data)
    - No need to cache processed data as csv (it will be cached when queries back from BigQuery during training)
    """
    min_date = parse(min_date).strftime('%Y-%m')
    max_date = parse(max_date).strftime('%Y-%m')
    pass


if __name__ == '__main__':
    preprocess()
