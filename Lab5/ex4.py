import os
from keras.datasets import mnist
import tensorflow as tf
import keras
from keras import layers
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

figures_directory = './figures'


def plot_images(name:str=None):
    indices = np.random.choice(X_test_norm.shape[0], size=5, replace=False)

    X_test_recon = model.predict(X_test_norm)
    X_test_noisy_recon = model.predict(X_test_noisy)

    fig, axes = plt.subplots(4, 5, figsize=(12, 8))

    for i, idx in enumerate(indices):
        # Original images
        axes[0, i].imshow(X_test_norm[idx, :, :, 0], cmap='gray')
        axes[0, i].axis('off')

        # Noisy images
        axes[1, i].imshow(X_test_noisy[idx, :, :, 0], cmap='gray')
        axes[1, i].axis('off')

        # Reconstruction from original
        axes[2, i].imshow(X_test_recon[idx, :, :, 0], cmap='gray')
        axes[2, i].axis('off')

        # Reconstruction from noisy
        axes[3, i].imshow(X_test_noisy_recon[idx, :, :, 0], cmap='gray')
        axes[3, i].axis('off')

    plt.tight_layout()
    if not name:
        plt.show()
    else:
        plt.savefig(f'{figures_directory}/{name}.pdf')


class Autoencoder(keras.Model):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = keras.Sequential([
            layers.Conv2D(8, (3, 3), activation='relu', strides=2, padding='same'),
            layers.Conv2D(4, (3, 3), activation='relu', strides=2, padding='same')
        ])

        self.decoder = keras.Sequential([
            layers.Conv2DTranspose(4, (3, 3), activation='relu', strides=2, padding='same'),
            layers.Conv2DTranspose(8, (3, 3), activation='relu', strides=2, padding='same'),
            layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


if __name__ == "__main__":
    if not os.path.isdir(figures_directory):
        os.makedirs(figures_directory, exist_ok=True)

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    X_train_norm = X_train / 255.0
    X_test_norm = X_test / 255.0

    X_train_norm_noise = X_train_norm + tf.random.normal(X_train.shape) * 0.35
    X_test_norm_noise = X_test_norm + tf.random.normal(X_test.shape) * 0.35

    X_train_noisy = tf.clip_by_value(X_train_norm_noise,clip_value_min=0, clip_value_max=1)
    X_test_noisy = tf.clip_by_value(X_test_norm_noise,clip_value_min=0, clip_value_max=1)

    model = Autoencoder()

    model.compile(
        optimizer='adam',
        loss='mse'
    )

    history = model.fit(
        X_train, X_train,
        epochs=25,
        batch_size=1024,
        validation_data=(X_test_noisy, X_test_noisy),
        verbose=2
    )

    X_train_pred = model.predict(X_train)
    train_errors = np.mean((X_train - X_train_pred) ** 2, axis=(1, 2, 3))
    threshold = np.mean(train_errors) + np.std(train_errors)
    print("Reconstruction error threshold:", threshold)

    X_test_pred = model.predict(X_test_norm)
    test_errors_clean = np.mean((X_test_norm - X_test_pred) ** 2, axis=(1, 2, 3))
    y_test_pred_clean = (test_errors_clean > threshold).astype(int)

    X_test_noisy_pred = model.predict(X_test_noisy)
    test_errors_noisy = np.mean((X_test_noisy - X_test_noisy_pred) ** 2, axis=(1, 2, 3))
    y_test_pred_noisy = (test_errors_noisy > threshold).astype(int)

    y_test_binary = (y_test != 0).astype(int)

    acc_clean = accuracy_score(y_test_binary, y_test_pred_clean)
    acc_noisy = accuracy_score(y_test_binary, y_test_pred_noisy)
    print("Accuracy on clean test images:", acc_clean)
    print("Accuracy on noisy test images:", acc_noisy)

    plot_images(name="plot_noisy_recon_modified_training")