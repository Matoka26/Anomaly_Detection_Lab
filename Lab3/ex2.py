from sklearn.datasets import make_blobs
from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.loda import LODA
import matplotlib.pyplot as plt
import numpy as np
import os

figures_directory = './figures'

if __name__ == "__main__":
    if not os.path.isdir(figures_directory):
        os.makedirs(figures_directory, exist_ok=True)

    centers = [(0, 10, 0), (10, 0, 10)]
    std = 1
    n_samples = 500
    n_features = 3

    x_samples, _ = make_blobs(
        n_samples=[n_samples, n_samples],
        n_features=n_features,
        centers=centers,
        cluster_std=std,
        random_state=42
    )

    loda = LODA(contamination=0.02)
    x_test = np.random.uniform(-10, 20, (1000, n_features))

    loda.fit(x_samples)

    scores = loda.decision_function(x_test)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
        x_test[:, 0], x_test[:, 1], x_test[:, 2],
        c=scores, cmap='inferno', s=25
    )

    fig.colorbar(scatter, ax=ax, label="Anomaly Score")
    ax.set_title("LODA 3D Anomaly Scores")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")

    plt.savefig(f"{figures_directory}/ex2_loda_3d_anomaly_scores.pdf")
