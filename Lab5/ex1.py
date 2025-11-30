import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

figures_directory = './figures'


def plot_samples_3d(samples: np.ndarray, show: bool=True) -> None:
    if not show:
        return

    x = samples[:, 0]
    y = samples[:, 1]
    z = samples[:, 2]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(x, y, z, s=20, alpha=0.6)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


def plot_cumsum_and_variance(eigs, name:str=None, show:bool=True) -> None:
    if not show:
        return

    sorted_eigs = np.sort(eigs)[::-1]

    total_var = np.sum(sorted_eigs)
    explained_var = sorted_eigs / total_var

    eig_cumsum = np.cumsum(explained_var)

    x = np.arange(1, len(eig_cumsum) + 1)

    plt.step(x, eig_cumsum, where="mid", label="Cumulative Explained Variance")
    plt.bar(x, explained_var, alpha=0.4, label="Individual Explained Variance")
    plt.legend()
    plt.grid(True)

    if not name:
        plt.show()
    else:
        plt.savefig(f'{figures_directory}/{name}.pdf')


def plot_over_each_pc(X_projected: np.ndarray, name:str=None, show:bool=True) -> None:
    if not show:
        return

    fig, ax = plt.subplots(nrows=1, ncols=X_projected.shape[1], figsize=(14, 6))
    for i in range(X_projected.shape[1]):
        pc = X_projected[:, i]
        pc_mean = np.mean(pc)
        pc_std = np.abs(pc - pc_mean)

        contamination = 0.1
        threshold = np.quantile(pc_std, 1 - contamination)

        labels = (pc_std > threshold).astype(int)

        ax[i].scatter(range(len(pc)), pc, c=labels)
        ax[i].axhline(threshold, linestyle='--', c='red')
        ax[i].axhline(-threshold, linestyle='--', label=f"Threshold:{threshold:.3f}", c='red',)
        ax[i].legend()
        ax[i].grid(True)
    if not name:
        plt.show()
    else:
        plt.savefig(f'{figures_directory}/{name}.pdf')


def plot_proj_new_space(X_projected: np.ndarray, name:str=None, show:bool=True) -> None:
    if not show:
        return

    centroid = np.mean(X_projected, axis=0)
    std_dev = np.std(X_projected, axis=0)
    z_scores = (X_projected - centroid) / std_dev
    distances = np.linalg.norm(z_scores, axis=1)

    contamination = 0.1
    threshold = np.quantile(distances, 1 - contamination)
    labels = (distances > threshold).astype(int)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot normal points
    ax.scatter(
        X_projected[labels == 0, 0],
        X_projected[labels == 0, 1],
        X_projected[labels == 0, 2],
        c='blue', alpha=0.6, label='Normal'
    )

    # Plot anomalies
    ax.scatter(
        X_projected[labels == 1, 0],
        X_projected[labels == 1, 1],
        X_projected[labels == 1, 2],
        c='red', alpha=0.8, label='Anomaly'
    )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend()
    if not name:
        plt.show()
    else:
        plt.savefig(f'{figures_directory}/{name}.pdf')


if __name__ == "__main__":
    if not os.path.isdir(figures_directory):
        os.makedirs(figures_directory, exist_ok=True)

    samples = np.random.multivariate_normal(
        mean=[5, 10, 2],
        cov=[[3, 2, 2], [2, 10, 1], [2, 1, 2]],
        size=500
    )

    plot_samples_3d(samples, show=False)

    samples_centered = samples - samples.mean(axis=0)
    cov_mat = np.cov(samples_centered, rowvar=False)
    eigs, vecs = np.linalg.eigh(cov_mat)

    plot_cumsum_and_variance(eigs, name="cumsum_and_variance", show=False)

    idx = np.argsort(eigs)[::-1]
    eigs_sorted = eigs[idx]
    vecs_sorted = vecs[:, idx]
    X_projected = samples_centered @ vecs_sorted

    plot_samples_3d(X_projected, show=False)

    plot_over_each_pc(X_projected, name="over_each_pc", show=False)

    plot_proj_new_space(X_projected, name="3d_projected")