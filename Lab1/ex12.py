import matplotlib.pyplot as plt
import numpy as np
from pyod.utils.data import generate_data
from pyod.models.knn import KNN
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc


if __name__ == "__main__":
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
