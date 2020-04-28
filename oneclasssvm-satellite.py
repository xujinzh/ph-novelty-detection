from sklearn.svm import OneClassSVM
import numpy as np
import pandas as pd
import copy
from tda import roc

head_train = 225
span = 100
tail_train = head_train + span
novelty = 75
normal = 150
test_num = novelty + normal
input_path = "./data/satellite-unsupervised-ad.csv"
satellite = pd.read_csv(input_path, header=None)
x_train = np.array(satellite.iloc[head_train:tail_train, :-1])
x_test = np.array(satellite.iloc[:test_num, :-1])
y_test = copy.deepcopy(satellite.iloc[:test_num, -1])
y_test.replace(['o', 'n'], [-1, 1], inplace=True)

clf = OneClassSVM(kernel="rbf", gamma='auto')
clf.fit(x_train)

predicted = clf.predict(x_test)
print(predicted)

y_scores = clf.score_samples(x_test)
print(y_scores)
roc.plot(y_test=np.array(y_test), y_scores=y_scores, pos_label=1, title='OCSVM - ')
