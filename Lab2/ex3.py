import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from pyod.models.knn import KNN
from pyod.models.lof import LOF
import os

figures_directory = './figures'

if __name__ == "__main__":
    if not os.path.isdir(figures_directory):
        os.makedirs(figures_directory, exist_ok=True)

    samples = np.concatenate((
            make_blobs(n_samples=200, n_features=2, center_box=(-10,-10), cluster_std=2)[0],
            make_blobs(n_samples=100, n_features=2, center_box=(10, 10), cluster_std=6)[0]
    ))

    knn_model = KNN(contamination=0.07, n_neighbors=5)
    knn_model.fit(samples)

    lof_model = LOF(contamination=0.07, n_neighbors=5)
    lof_model.fit(samples)

    knn_pred = knn_model.predict(samples)
    lof_pred = lof_model.predict(samples)

    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(8, 4))
    ax[0].scatter(samples[:, 0], samples[:, 1], c=knn_pred)
    ax[0].set_title("KNN Model")
    ax[1].scatter(samples[:, 0], samples[:, 1], c=lof_pred)
    ax[1].set_title("LOF Model")
    ax[0].grid(True)
    ax[1].grid(True)
    plt.savefig("./figures/ex3_knn_vs_lof.pdf")

