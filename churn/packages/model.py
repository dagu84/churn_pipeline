import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

def initialise_model():
    model = LogisticRegression(penalty='elasticnet', l1_ratio=0.3, solver='saga')
    return model

def evaluate_model(model, X: np.ndarray, y: np.ndarray):
    cv = cross_validate(model, scoring=['accuracy'], cv=5, n_jobs=-1)
    accuracy = cv['test_accuracy'].max()

    print(f"Model evaluated, accuracy: {round(accuracy, 2)}")

    return accuracy
