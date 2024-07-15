############### SET-UP ###############
install:
	@pip install -r requirements.txt

############### STEPS ###############
new_data:
	python -c 'from churn.main import data_pipeline; data_pipeline()'

prepare_data:
	python -c 'from churn.main import preprocess; preprocess()'

train_model:
	python -c 'from churn.main import train; train()'

evaluate_model:
	python -c 'from churn.main import evaluate; evaluate()'

############### TESTS ###############
