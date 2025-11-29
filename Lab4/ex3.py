import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pyod.models.ocsvm import OCSVM
from pyod.models.deep_svdd import DeepSVDD
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

data_path = "./data/shuttle 1.mat"


def balanced_accuracy_pyod_format(y_true, y_pred):
    # convert (-1 anomaly, +1 normal) → (1 anomaly, 0 normal)
    y_pred = np.where(y_pred == -1, 1, 0)
    return balanced_accuracy_score(y_true, y_pred)


if __name__ == "__main__":
    data = scipy.io.loadmat(data_path)
    X = data["X"]
    y = data["y"].ravel()


    X_train, X_test, _, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42, stratify=y
    )


    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_ocsvm = OCSVM()
    model_ocsvm.fit(X_train_scaled)
    y_pred_ocsvm = np.where(model_ocsvm.predict(X_test_scaled) == -1, 1, 0)
    ocsvm_score = model_ocsvm.decision_function(X_test_scaled)

    print("~~~ OCSVM ~~~")
    print(f"\tBalanced Accuracy: {balanced_accuracy_score(y_test, y_pred_ocsvm):.4f}")
    print(f"\tArea Under Curve: {roc_auc_score(y_test, ocsvm_score):.4f}\n")

    architectures = {
        "shallow": [32, 16],
        "medium": [64, 32],
        "deep": [128, 64, 32]
    }

    for name, layers in architectures.items():
        print(f"~~~ DeepSVDD ({name}) ~~~")

        model_deepsvdd = DeepSVDD(
            n_features=X_train_scaled.shape[1],
            hidden_neurons=layers
        )
        model_deepsvdd.fit(X_train_scaled)

        y_pred = np.where(model_deepsvdd.predict(X_test_scaled) == -1, 1, 0)
        score = model_deepsvdd.decision_function(X_test_scaled)

        bal_acc = balanced_accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, score)

        print(f"\tBalanced Accuracy: {bal_acc:.4f}")
        print(f"\tArea Under Curve: {auc:.4f}\n")
