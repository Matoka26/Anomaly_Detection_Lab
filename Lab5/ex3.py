import os
import matplotlib.pyplot as plt
import keras
import scipy.io
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras import layers
from sklearn.metrics import balanced_accuracy_score

data_path = "./data/shuttle.mat"
figures_directory = './figures'


class Autoencoder(keras.Model):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()

        self.encoder = keras.Sequential([
            layers.Dense(8, activation='relu', input_shape=(input_dim,)),
            layers.Dense(5, activation='relu'),
            layers.Dense(3, activation='relu')
        ])

        self.decoder = keras.Sequential([
            layers.Dense(5, activation='relu', input_shape=(3,)),
            layers.Dense(8, activation='relu'),
            layers.Dense(9, activation='sigmoid')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


if __name__ == "__main__":
    if not os.path.isdir(figures_directory):
        os.makedirs(figures_directory, exist_ok=True)

    data = scipy.io.loadmat(data_path)
    X = data["X"]
    y = data["y"].ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42, stratify=y
    )

    scalar = MinMaxScaler()
    X_train_scaled = scalar.fit_transform(X=X_train, y=[0, 1])
    X_test_scaled = scalar.fit_transform(X=X_test, y=[0, 1])

    input_dim = X.shape[1]
    model = Autoencoder(input_dim)

    model.compile(
        optimizer='adam',
        loss='mse'
    )

    history = model.fit(
        X_train_scaled, X_train_scaled,
        epochs=100,
        batch_size=1024,
        validation_data=(X_test_scaled, X_test_scaled),
        verbose=2
    )

    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.legend()
    # plt.savefig(f'{figures_directory}/train_and_validation_loss.pdf')
    plt.show()

    X_train_pred = model.predict(X_train_scaled)
    X_test_pred = model.predict(X_test_scaled)

    train_errors = np.mean((X_train_scaled - X_train_pred) ** 2, axis=1)
    test_errors = np.mean((X_test_scaled - X_test_pred) ** 2, axis=1)
    contamination = np.mean(y_train != 0)
    threshold = np.quantile(train_errors, 1 - contamination)
    print("Reconstruction error threshold:", threshold)

    y_train_pred = (train_errors > threshold).astype(int)
    y_test_pred = (test_errors > threshold).astype(int)

    bal_acc_train = balanced_accuracy_score(y_train != 0, y_train_pred)
    bal_acc_test = balanced_accuracy_score(y_test != 0, y_test_pred)

    print(f"Balanced Accuracy - Train: {bal_acc_train:.4f}")
    print(f"Balanced Accuracy - Test:  {bal_acc_test:.4f}")