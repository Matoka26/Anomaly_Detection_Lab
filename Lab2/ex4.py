import numpy as np
from pyod.models.knn import KNN
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import balanced_accuracy_score
from pyod.utils.utility import standardizer
from pyod.models.combination import maximization, average
import os

figures_directory = './figures'
data_path = './data/cardio.mat'

if __name__ == "__main__":
    if not os.path.isdir(figures_directory):
        os.makedirs(figures_directory, exist_ok=True)

    data = loadmat(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        data["X"], data["y"], test_size=0.33, random_state=42
    )

    X_train = MinMaxScaler().fit_transform(X_train)
    X_test = MinMaxScaler().fit_transform(X_test)

    n_neighbours = np.arange(30, 120, 9)
    knn_models = []
    y_train_pred = []
    y_test_pred = []
    train_scores = []
    test_scores = []

    for nn in n_neighbours:
        model = KNN(n_neighbors=nn)
        model.fit(X_train)
        knn_models.append(model)

        y_train_pred.append(model.predict(X_train))
        y_test_pred.append(model.predict(X_test))

        train_scores.append(model.decision_scores_)
        test_scores.append(model.decision_function(X_test))

        print(f"N:{nn} BA TRAIN: {balanced_accuracy_score(y_train, y_train_pred[-1]):.2f}")
        print(f"N:{nn} BA TEST : {balanced_accuracy_score(y_test, y_test_pred[-1]):.2f}\n")

    train_scores = np.array(train_scores).T
    test_scores = np.array(test_scores).T

    train_scores_norm = standardizer(train_scores)
    test_scores_norm = standardizer(test_scores)

    train_max = maximization(train_scores_norm)
    test_max = maximization(test_scores_norm)

    train_avg = average(train_scores_norm)
    test_avg = average(test_scores_norm)

    train_max_threshold = np.quantile(train_max, q=0.80)
    test_max_threshold = np.quantile(test_max, q=0.80)

    train_avg_threshold = np.quantile(train_avg, q=0.80)
    test_avg_threshold = np.quantile(test_avg, q=0.80)

    max_ensemble_train_pred = train_max >= train_max_threshold
    max_ensemble_test_pred = test_max >= test_max_threshold

    avg_ensemble_train_pred = train_avg >= train_avg_threshold
    avg_ensemble_test_pred = test_avg >= test_avg_threshold

    print(f"\n\nTRAIN BA with Averaging {balanced_accuracy_score(avg_ensemble_train_pred, y_train):.4f}")
    print(f"TRAIN BA with Maximization {balanced_accuracy_score(max_ensemble_train_pred, y_train):.4f}")

    print(f"TEST BA with Averaging {balanced_accuracy_score(avg_ensemble_test_pred, y_test):.4f}")
    print(f"TEST BA with Maximization {balanced_accuracy_score(max_ensemble_test_pred, y_test):.4f}")
