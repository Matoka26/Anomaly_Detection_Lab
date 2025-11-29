import scipy.io
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from itertools import product
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score, make_scorer

data_path = "./data/cardio 1.mat"


def ocsvm_balanced_accuracy(y_true, y_pred):
    # convert (-1 anomaly, +1 normal) → (1 anomaly, 0 normal)
    y_pred = np.where(y_pred == -1, 1, 0)
    return balanced_accuracy_score(y_true, y_pred)


if __name__ == "__main__":

    df = scipy.io.loadmat(data_path)
    X = df["X"]
    y = df["y"].ravel()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.6, random_state=42, stratify=y
    )

    param_grid = {
        "ocsvm__kernel": ["rbf", "poly", "sigmoid"],
        "ocsvm__gamma": ["scale", "auto", 0.01, 0.1, 1],
        "ocsvm__nu": [0.01, 0.05, 0.1, 0.2, 0.4]
    }

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ("ocsvm", OneClassSVM())
    ])

    balanced_acc_scorer = make_scorer(ocsvm_balanced_accuracy)

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=balanced_acc_scorer,
        cv=5,  # 5-fold CV on the training set
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    print("Best parameters:", grid.best_params_)
    print("Best CV Balanced Accuracy:", grid.best_score_)