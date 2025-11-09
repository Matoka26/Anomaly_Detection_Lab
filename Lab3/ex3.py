from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.loda import LODA
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import warnings

warnings.filterwarnings("ignore", message=".*pin_memory.*")

figures_directory = './figures'

if __name__ == "__main__":
    if not os.path.isdir(figures_directory):
        os.makedirs(figures_directory, exist_ok=True)

    statlog_shuttle = fetch_ucirepo(id=148)  # UCI Statlog (Shuttle)
    X = statlog_shuttle.data.features.to_numpy()
    y = statlog_shuttle.data.targets.to_numpy().ravel()

    print(f"Original shape: X={X.shape}, y={y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "iforest" : IForest(),
        "dif"     : DIF(),
        "loda"    : LODA()
    }

    y_train_binary = (y_train != 1).astype(int)
    y_test_binary = (y_test != 1).astype(int)

    roc_scores = []
    ba_scores = []
    for name, model in models.items():
        model.fit(X_train_scaled)

        y_pred = model.predict(X_test_scaled)
        scores = model.decision_function(X_test_scaled)

        ba = balanced_accuracy_score(y_test_binary, y_pred)
        roc = roc_auc_score(y_test_binary, scores)

        ba_scores.append(ba)
        roc_scores.append(roc)

        print(f"{name}: Balanced Accuracy = {ba:.4f}, ROC AUC = {roc:.4f}")

    print(f"\nMean Balanced Accuracy = {np.mean(ba_scores):.4f}")
    print(f"Mean ROC AUC = {np.mean(roc_scores):.4f}")
