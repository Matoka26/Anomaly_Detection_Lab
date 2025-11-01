import numpy as np
import matplotlib.pyplot as plt
import os

figures_directory = './figures'

if __name__ == "__main__":
    if not os.path.isdir(figures_directory):
        os.makedirs(figures_directory, exist_ok=True)

    n = 30
    a = 1
    b = 1

    sigmas = [1, 5]

    samples_x = []
    samples_y = []
    i = 0
    for var_x in sigmas:
        for var_y in sigmas:
            x = np.random.normal(0, var_x, n)
            eps = np.random.normal(0, var_y, n)
            y = a * x + b + eps

            samples_y.append(y)
            samples_x.append(x)

    X = np.column_stack([x, np.ones_like(x)])
    U, _, _ = np.linalg.svd(X, full_matrices=False)
    H = U @ U.T
    top_k = np.argsort(np.diag(H))[-20:]
    plt.scatter(samples_x, samples_y, color='b', label="Low Leverage", alpha=0.5)
    plt.scatter(x[top_k], y[top_k], color="r", label="High Leverage")

    plt.tight_layout()
    plt.grid(True)
    plt.savefig("./figures/ex1.pdf")