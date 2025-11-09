from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import os

figures_directory = './figures'

if __name__ == "__main__":
    if not os.path.isdir(figures_directory):
        os.makedirs(figures_directory, exist_ok=True)

    n_feats = 2
    n_projections = 5
    n_bins = 20
    n_samples = 500

    samples, _ = make_blobs(n_samples=n_samples, n_features=n_feats)
    proj_vecs = np.random.multivariate_normal(mean=[0] * n_feats, cov=np.eye(n_feats), size=n_projections)
    proj_vecs /= np.linalg.norm(proj_vecs, axis=1, keepdims=True)

    projections = samples @ proj_vecs.T

    probs = np.zeros_like(projections)
    hist_params = []
    for i in range(n_projections):
        p = projections[:, i]

        data_range = (p.min() - 1, p.max() + 1)
        counts, bin_edges = np.histogram(p, bins=n_bins, range=data_range, density=False)

        bin_probs = counts / counts.sum()

        bin_indices = np.digitize(p, bin_edges) - 1

        bin_indices = np.clip(bin_indices, 0, len(bin_probs) - 1)
        probs[:, i] = bin_probs[bin_indices]
        hist_params.append((bin_edges, bin_probs))

    anomaly_scores = probs.mean(axis=1)

    # print(f"Anomaly scores:{anomaly_scores}")

    x_test = np.random.uniform(-3, 3, (500, 2))
    y_test = x_test @ proj_vecs.T

    test_probs = np.zeros_like(y_test)
    for i in range(n_projections):
        bin_edges, bin_probs = hist_params[i]
        bin_indices = np.digitize(y_test[:, i], bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, len(bin_probs) - 1)
        test_probs[:, i] = bin_probs[bin_indices]

    anomaly_scores = test_probs.mean(axis=1)

    plt.figure(figsize=(6, 5))
    plt.scatter(x_test[:, 0], x_test[:, 1], c=anomaly_scores, cmap="inferno", s=30)
    plt.colorbar(label="Anomaly Score")
    # plt.grid(True)
    plt.savefig(f"{figures_directory}/ex1.pdf")