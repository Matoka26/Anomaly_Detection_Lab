from pyod.utils.data import generate_data
from pyod.models.ocsvm import OCSVM
from pyod.models.deep_svdd import DeepSVDD
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import os

figures_directory = './figures'

# Helper scatter function
def plot_3d(ax, X, y, title):
    ax.scatter(X[y == 0, 0], X[y == 0, 1], X[y == 0, 2],
               c='blue', s=20, label="Inliers")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], X[y == 1, 2],
               c='red', s=20, label="Outliers")
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_zlabel("Feature 3")
    ax.legend()


if __name__ == "__main__":
    os.makedirs(figures_directory, exist_ok=True)

    contamination = 0.15
    X_train, X_test, y_train, y_test = generate_data(
        n_train=300,
        n_test=200,
        n_features=3,
        contamination=0.15,
        random_state=42
    )

    model = DeepSVDD(contamination=contamination, epochs=20, n_features=3)
    model.fit(X_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    y_scores = model.decision_function(X_test)
    bal_acc = balanced_accuracy_score(y_test, y_pred_test)
    roc = roc_auc_score(y_test, y_scores)

    print("Balanced Accuracy:", bal_acc)
    print("ROC AUC:", roc)

    fig = plt.figure(figsize=(8, 7))

    ax1 = fig.add_subplot(221, projection='3d')
    plot_3d(ax1, X_train, y_train, "Training Data (Ground Truth)")

    ax2 = fig.add_subplot(222, projection='3d')
    plot_3d(ax2, X_test, y_test, "Test Data (Ground Truth)")

    ax3 = fig.add_subplot(223, projection='3d')
    plot_3d(ax3, X_train, y_pred_train, "Training Data (Predicted Labels)")

    ax4 = fig.add_subplot(224, projection='3d')
    plot_3d(ax4, X_test, y_pred_test, "Test Data (Predicted Labels)")

    plt.tight_layout()
    plt.savefig(f'{figures_directory}/ex1_deepsvd.pdf')
