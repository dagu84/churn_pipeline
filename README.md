Created a fully automated churn modelling pipeline.

Data: Hosted on BigQuery, and locally cached
  - original dataset was downloaded from Kaggle
  - data was cleaned and hosted on BigQuery
  - once new data has been generated (custom function), it will be injested into BQ after cleaning
  - the data is then preprocessed (feaure engineering + general preprocessing) and injested into another BQ table

Model: Hosted on MLflow
  - once the new data has been injested into BQ the current 'production' model will be tested on the data through CV
  - the results of the test will be stored alongside the model in MLflow
  - a new model will be trained and tested on the new data, with the evaluation metrics being compared between the two
  - the higher perfoming model will either remain as the 'production' model or will be staged into production

Orchestrated: Prefect
  - prefect workflow orchestrates the whole process in the order inteded on the workflow.py file
