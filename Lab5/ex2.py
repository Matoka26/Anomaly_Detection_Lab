import scipy.io
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pyod.models.pca import PCA
from pyod.models.kpca import KPCA
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score

data_path = "./data/shuttle 1.mat"
figures_directory = './figures'

if __name__ == "__main__":
    if not os.path.isdir(figures_directory):
        os.makedirs(figures_directory, exist_ok=True)

    data = scipy.io.loadmat(data_path)
    X = data["X"]
    y = data["y"].ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.6, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = PCA()
    model.fit(X_train_scaled)

    kpca = KPCA()
    kpca.fit(X_train_scaled)

    explained_var = model.explained_variance_
    cum_explained_var = np.cumsum(explained_var) / np.sum(explained_var)

    plt.step(range(len(cum_explained_var)), cum_explained_var, where='mid', label='Cumulative')
    plt.bar(range(len(explained_var)), explained_var / np.sum(explained_var), alpha=0.4, label='Individual')
    plt.grid(True)
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance")
    plt.legend()
    # plt.savefig(f'{figures_directory}/pyod_pca.pdf')
    # plt.show()

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    bal_acc_train = balanced_accuracy_score(y_train != 0, y_train_pred)
    bal_acc_test = balanced_accuracy_score(y_test != 0, y_test_pred)
    print(f"PCA: Balanced Accuracy - Train: {bal_acc_train:.4f}")
    print(f"PCA:  Balanced Accuracy - Test:  {bal_acc_test:.4f}")

    y_train_pred = kpca.predict(X_train_scaled)
    y_test_pred = kpca.predict(X_test_scaled)

    bal_acc_train = balanced_accuracy_score(y_train != 0, y_train_pred)
    bal_acc_test = balanced_accuracy_score(y_test != 0, y_test_pred)
    print(f"KPCA: Balanced Accuracy - Train: {bal_acc_train:.4f}")
    print(f"KPCA:  Balanced Accuracy - Test:  {bal_acc_test:.4f}")