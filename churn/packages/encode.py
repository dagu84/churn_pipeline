import pandas as pd
from churn.packages.params import *


def binary_encode(series: pd.Series) -> pd.Series:
    return series.apply(lambda x: 0 if x == 'No' else 1)


def cat_encode(series: pd.Series) -> pd.Series:
    series.apply(lambda x: 'No' if x != 'Yes' else 'Yes')
    return series.apply(lambda x: 0 if x == 'No' else 1)
