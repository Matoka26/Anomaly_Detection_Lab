import matplotlib.pyplot as plt
import numpy as np
from pyod.utils.data import generate_data
from pyod.models.knn import KNN
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

def ex1_2():
   contamination_rate = 0.25

   X_train, X_test, Y_train, Y_test = generate_data(n_train=500, n_test=100, n_features=2, contamination=contamination_rate)

   plt.scatter(X_train[Y_train == 0, 0], X_train[Y_train == 0, 1], c='b', label="inliners")
   plt.scatter(X_train[Y_train == 1, 0], X_train[Y_train == 1, 1], c='r', label="outliners")
   plt.grid(True)
   plt.legend()
   plt.show()

   # train kNN detector
   clf_name = 'KNN'
   clf = KNN()
   clf.fit(X_train)

   # get the prediction labels and outlier scores of the training data
   Y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)

   # get the prediction on the test data
   Y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)

   tn, fp, fn, tp = confusion_matrix(Y_test_pred, Y_test).ravel().tolist()

   print(f'TN:{tn}\nFP:{fp}\nFN:{fn}\nTP:{tp}')
   print(f'Balance Accuracy: {(tp/(tp+fn) + tn/(tn+fp))/2}%')

   fpr, tpr, thresholds = roc_curve(Y_test, Y_test_pred)
   roc_auc = auc(fpr, tpr)

   plt.figure()
   plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
   plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
   plt.xlim([0.0, 1.0])
   plt.ylim([0.0, 1.05])
   plt.xlabel('False Positive Rate')
   plt.ylabel('True Positive Rate')
   plt.title('ROC Curve for Breast Cancer Classification')
   plt.legend()
   plt.grid(True)
   plt.show()


def ex3():
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

def ex4():
   d = 3

   miu = np.random.rand(d)
   l = np.random.rand(d)
   covariance = np.outer(l, l)
   print('Mean:', miu)
   print('Covariance\n', covariance)

   x = np.random.randn(1000, d)

   y = l * x + miu

   # https://medium.com/@whyamit101/understanding-z-score-with-numpy-bc8b23f81639
   mean_y = np.mean(y, axis=0)
   std_y = np.std(y, axis=0)

   z_scores = (y - mean_y) / std_y

   print('Z-scores\n', z_scores)

   contamination_rate = 0.1
   threshold = np.quantile(np.abs(z_scores), 1 - contamination_rate)

   y_pred = np.abs(z_scores).max(axis=1) > threshold

   Y = np.zeros(1000)
   Y[np.random.choice(1000, size=int(1000 * contamination_rate), replace=False)] = 1  # Randomly assign anomalies

   tn, fp, fn, tp = confusion_matrix(Y, y_pred).ravel()

   print(f'TN: {tn}\nFP: {fp}\nFN: {fn}\nTP: {tp}')

   # Balanced accuracy calculation
   balanced_accuracy = (tp / (tp + fn) + tn / (tn + fp)) / 2
   print(f'Balanced Accuracy: {balanced_accuracy * 100:.2f}%')

if __name__ == "__main__":
   # ex1_2()
   # ex3()
   ex4()