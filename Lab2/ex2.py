import numpy as np
import matplotlib.pyplot as plt
from pyod.models.knn import KNN
from pyod.utils.data import generate_data_clusters
from sklearn.metrics import balanced_accuracy_score
import os

figures_directory = './figures'

if __name__ == "__main__":
    if not os.path.isdir(figures_directory):
        os.makedirs(figures_directory, exist_ok=True)

    X_train, X_test, y_train, y_test = generate_data_clusters(
        n_train=4000, n_test=2000,
        n_clusters=2, n_features=2,
        contamination=0.1
    )

    n_neighbours = [1, 3, 7, 10]

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(6, 4))
    ax = ax.flatten()
    for i, neigh in enumerate(n_neighbours):
        model = KNN(n_neighbors=neigh)
        model.fit(X_train)
        y_pred = model.predict(X_test)

        ax[i].scatter(X_test[:, 0], X_test[:, 1], c=y_pred)
        ax[i].grid(True)
        ax[i].set_title(f'N = {neigh}, BA = {balanced_accuracy_score(y_test, y_pred):.4f}')
    plt.tight_layout()
    plt.savefig("./figures/ex2_n_values_comparisson.pdf")

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(6, 4))
    ax = ax.flatten()

    ax[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train)
    ax[0].set_title("Ground truth labels for training data")

    ax[2].scatter(X_test[:, 0], X_test[:, 1], c=y_test)
    ax[2].set_title("Ground truth labels for test data")

    ax[3].scatter(X_test[:, 0], X_test[:, 1], c=y_pred)
    ax[3].set_title("Predicted labels for test data")

    y_pred = model.predict(X_train)
    ax[1].scatter(X_train[:, 0], X_train[:, 1], c=y_pred)
    ax[1].set_title("Predicted labels for train data")

    for i in range(0, 4):
        ax[i].grid(True)
    plt.tight_layout()
    plt.savefig("./figures/ex2_predictionsq.pdf")

