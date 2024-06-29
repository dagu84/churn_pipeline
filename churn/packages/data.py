import pandas as pd

from churn.packages.params import *

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the data.
    - Removes null values
    - Removes duplicates
    - Checks for other empty values such as ""
    """

    # Drop null values
    df = df.dropna()

    # Drop duplicates
    df = df.drop_duplicates()

    # Drop the TotalCharges column
    df = df.drop(columns=['TotalCharges'])

    return df
