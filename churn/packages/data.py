import numpy as np
import pandas as pd
from pathlib import Path
from colorama import Fore, Style
from google.cloud import bigquery

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
    df = df.drop(columns=[['TotalCharges', 'Date']])

    return df


def get_data_with_cache(
        gcp_project:str,
        query:str,
        cache_path:Path,
        data_has_header=True
    ) -> pd.DataFrame:
    """
    Retrieve 'query' data from BigQuery, or from 'cache_path' if the file exists
    Store at cache_path if retrieved from BigQuery for future use
    """
    if cache_path.is_file():
        print(Fore.BLUE + "\nLoad data from local CSV..." + Style.RESET_ALL)
        df = pd.read_csv(cache_path, header='infer' if data_has_header else None)
    else:
        print(Fore.BLUE + "\nLoad data from BigQuery server..." + Style.RESET_ALL)
        client = bigquery.Client(project=gcp_project)
        query_job = client.query(query)
        results = query_job.results()
        df = results.to_dataframe()

        if df.shape[0] > 1:
            df.to_csv(cache_path, header=data_has_header, index=False)

    print(f"✅ Data loaded, with shape {df.shape}")

    return df


def load_data_to_bq(
        data: pd.DataFrame,
        gcp_project: str,
        bq_dataset: str,
        table: str,
        truncate: bool
    ) -> None:
    """
    - Save the DataFrame to BigQuery
    - Empty the table beforehand if 'truncate' is True, append otherwise
    """
    assert isinstance(data, pd.DataFrame)
    full_table_name = f"{gcp_project}.{bq_dataset}.{table}"
    print(Fore.BLUE + f"\nSave data to BigQuery @ {full_table_name}...:" + Style.RESET_ALL)
    data.columns = [f"_{column}" if not str(column)[0].isalpha() and not str(column)[0] == "_" else str(column) for column in data.columns]

    client = bigquery.Client()

    write_mode = "WRITE_TRUNCATE" if truncate else "WRITE_APPEND"
    job_config = bigquery.LoadJobConfig(write_disposition=write_mode)

    print(f"\n{'Write' if truncate else 'Append'} {full_table_name} ({data.shape[0]} rows)")

    job = client.load_table_from_dataframe(data, full_table_name, job_config=job_config)
    result = job.result()

    print(f"✅ Data saved to bigquery, with shape {data.shape}")


def create_data(source_file: Path, DATE) -> pd.DataFrame:
    """
    - Creates an artificial dataset based on a source csv as input
    - Edits the date column to reflect the '%Y-%m' of the current date
    """
    num_rows = 2000
    df = pd.read_csv(source_file)

    def generate_artifical_data(df, num_rows):
        artificial_data = pd.DataFrame()

        for column in df.columns:
            if df[column].dtype == 'object':
                artificial_data[column] = np.random.choice(df[column].unique(), num_rows)
        else:
            artificial_data[column] = np.random.choice(df[column].values, num_rows)

        return artificial_data

    artificial_data = generate_artifical_data(df, num_rows)

    format_date = DATE.strftime('%Y-%m')
    artificial_data['Date'] = format_date

    return artificial_data
