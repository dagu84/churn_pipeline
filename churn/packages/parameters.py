import os
import datetime

####### CREDENTIAlS #######
# ChatGPT
GPT_API_KEY = os.environ.get("GPT_API_KEY")
GPT_ORG_ID = os.environ.get("GPT_ORG_ID")

# GCP
GOOGLE_APPLICATION_CREDENTIALS = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
PROJECT_ID = os.environ.get("PROJECT_ID")
BQ_DATASET = os.environ.get("BQ_DATASET")
RAW_TABLE = os.environ.get("RAW_TABLE")
PROCESSED_TABLE = os.environ.get("PROCESSED_TABLE")

#MLflow
MODEL_NAME = os.environ.get("MODEL_NAME")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MFLOW_EXPERIEMENT = os.environ.get("MFLOW_EXPERIEMENT")

# Prefect
API_KEY = os.environ.get("API_KEY")
PREFECT_FLOW_NAME = os.environ.get("PREFECT_FLOW_NAME")
PREFECT_LOG_LEVEL = os.environ.get("PREFECT_LOG_LEVEL")

####### CONTSANTS #######
COLUMN_NAME_ROWS = ['customerID', 'Date', 'gender', 'SeniorCitizen', 'Partner',
       'Dependents', 'tenure', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
       'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges',
       'Churn']

CURRENT_DATE = datetime.date.today()
LOCAL_PATH = os.environ.get("LOCAL_PATH")
