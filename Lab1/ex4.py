import matplotlib.pyplot as plt
import numpy as np
from pyod.utils.data import generate_data
from pyod.models.knn import KNN
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

if __name__ == "__main__":
   d = 3

   miu = np.random.random((d, 1))
   L = np.random.normal(0, 1, size=(d, d))
   E = L @ L.T
   C = np.linalg.cholesky(E)

   x = np.random.normal(0, 1, size=(d, d))
   y = C @ x + miu

   dif = y - miu
   sol = np.linalg.solve(E, dif)

   z_scores = (y - miu).T @ sol
   quant_90 = np.quantile(z_scores, 0.9, axis=0)

   is_anomaly = z_scores > quant_90

   y_pred = is_anomaly.astype(int).flatten()

   print(x)
   # Dont have a ground truth to compute BA and conf matrix