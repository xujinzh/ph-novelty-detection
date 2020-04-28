from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import pandas as pd
from tda import roc
import copy

head_train = 2000  # the index of first novelty point
span = 100  # the number of train set, and is ratio time of the train set in ph-breast.py
tail_train = head_train + span
novelty = 1508
normal = 1508
test_num = novelty + normal  # 10 novelty point and 20 normal point, similar t o ph-breast.py
input_path = "./data/aloi-unsupervised-ad.csv"
data = pd.read_csv(input_path, header=None)
x_train = np.array(data.iloc[head_train:tail_train, :-1])
x_test = np.array(data.iloc[:test_num, :-1])
y_test = copy.deepcopy(data.iloc[:test_num, -1])
y_test.replace(['o', 'n'], [-1, 1], inplace=True)

clf = LocalOutlierFactor(novelty=True, n_neighbors=9)

T = clf.fit(x_train)
print("threshold: ", T)

predicted = clf.predict(x_test)
print(predicted)

y_scores = clf.score_samples(x_test)
print(y_scores)

roc.plot(y_test=np.array(y_test), y_scores=y_scores, pos_label=1, title='LOF - ')
