import pandas as pd
import numpy as np
from colorama import Fore, Style
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

from churn.packages.encode import encode
from churn.packages.params import *

def preprocess_features(X: pd.DataFrame) -> np.ndarray:
    def create_pipeline() -> ColumnTransformer:
        """
        Scikit-Learn pipeline that transformes a cleaned dataset into a preprocessed array

        Stateless operation
        """

        # Continuous features
        monthly_min = X['MonthlyCharges'].min()
        monthly_max = X['MonthlyCharges'].max()
        monthly_pipe = make_pipeline(FunctionTransformer(lambda x: (x - monthly_min) / (monthly_max - monthly_min), validate=True))

        tenure_min = X['tenure'].min()
        tenure_max = X['tenure'].max()
        tenure_pipe = make_pipeline(FunctionTransformer(lambda x: (x - tenure_min) / (tenure_max - tenure_min), validate=True))

        # Categorical features
        cat_pipe = make_pipeline(FunctionTransformer(encode, validate=True))

        # Multi-cat features
        multi_pipe = make_pipeline(OneHotEncoder(handle_unknown='ignore'))

        # Pipeline
        pipeline = ColumnTransformer(
            [
            ("monthly_pipe", monthly_pipe, ['MonthlyCharges']),
            ("tenure_pipe", tenure_pipe, ['tenure']),
            ("cat_pipe", cat_pipe, ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']),
            ("multi_pipe", multi_pipe, ['Contract', 'PaymentMethod', 'InternetService'])
            ],
            n_jobs=-1, remainder='passthrough'
        )

        return pipeline

    print(Fore.BLUE + "\nPreprocessing features..." + Style.RESET_ALL)

    pipeline = create_pipeline()
    X_processed = pipeline.fit_transform(X)
    print("✅ X_processed, with shape", X_processed.shape)

    return X_processed
