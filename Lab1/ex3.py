import matplotlib.pyplot as plt
import numpy as np
from pyod.utils.data import generate_data
from pyod.models.knn import KNN
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc


if __name__ == "__main__":
   contamination_rate = 0.1
   X, _, Y, _ = generate_data(n_train=1000, n_test=0, n_features=1, contamination=contamination_rate)
   mean = np.mean(X)
   std = np.std(X)

   def z_score(x):
      return np.abs(x-mean) / std


   # Compute the threshold based on the contamination rate
   threshold = np.quantile([z_score(x) for x in X], 1 - contamination_rate)

   # Predict anomalies: Z-scores greater than threshold are considered anomalies
   y_pred = [1 if z_score(sample) > threshold else 0 for sample in X]

   tn, fp, fn, tp = confusion_matrix(y_pred, Y).ravel().tolist()

   print(f'TN:{tn}\nFP:{fp}\nFN:{fn}\nTP:{tp}')
   print(f'Balance Accuracy: {(tp / (tp + fn) + tn / (tn + fp)) / 2}%')
