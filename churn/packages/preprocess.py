import pandas as pd
import numpy as np
from colorama import Fore, Style
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

from churn.packages.encode import binary_encode, cat_encode
from churn.packages.params import *

def preprocessor(X: pd.DataFrame) -> np.ndarray:
    def create_pipeline() -> ColumnTransformer:
        pass
