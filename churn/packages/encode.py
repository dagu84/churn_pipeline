import pandas as pd
import numpy as np
from packages.params import *

def binary_encode(series: pd.DataFrame) -> np.ndarray:
    series = pd.DataFrame(series)
    for i in series:
        series[i] = series[i].astype(int)
    return series.values

def cat_encode(series: pd.Series) -> pd.Series:
    if isinstance(series, np.ndarray):
        series = pd.Series(series)
    series = series.apply(lambda x: 'No' if x != 'Yes' else 'Yes')
    return series.apply(lambda x: 0 if x == 'No' else 1)

def encode(series: pd.DataFrame) -> np.ndarray:
    series = pd.DataFrame(series)
    for i in series:
       series[i] = series[i].apply(lambda x: 0 if x != 'Yes' else 1)
    return series.values
