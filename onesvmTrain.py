from sklearn.svm import OneClassSVM
import numpy as np
import pandas as pd
import copy
from tda import roc

outlier_number = 225
span = 100
test_num = 75 + 150
input_path = "./data/satellite-unsupervised-ad.csv"
satellite = pd.read_csv(input_path, header=None)
x_train = np.array(satellite.iloc[outlier_number:outlier_number + span, :-1])
x_test = np.array(satellite.iloc[:test_num, :-1])
y_test = copy.deepcopy(satellite.iloc[:test_num, -1])
y_test.replace(['o', 'n'], [-1, 1], inplace=True)

clf = OneClassSVM(kernel="rbf", gamma='auto')
clf.fit(x_train)

predicted = clf.predict(x_test)
print(predicted)

y_scores = clf.score_samples(x_test)
print(y_scores)
roc.plot(y_test=np.array(y_test), y_scores=y_scores, pos_label=1)
