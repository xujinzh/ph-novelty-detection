import numpy as np
import pandas as pd
from tda import topology as top
from tda import roc
import copy

head_train = 15  # the index of first novelty point
span = 100  # the number of train set, and the ratio * span is the base shape data set
tail_train = head_train + span
base_lower = 20  # the minimum of the points in base shape data set
threshold = 1.0
novelty = 10
normal = 40
test_num = novelty + normal  # 10 novelty point and 20 normal point
input_path = "./data/breast-cancer-unsupervised-ad.csv"
data = pd.read_csv(input_path, header=None)
x_train = np.array(data.iloc[head_train:tail_train, :-1])
x_test = np.array(data.iloc[:test_num, :-1])
y_test = copy.deepcopy(data.iloc[:test_num, -1])
y_test.replace(['o', 'n'], [-1, 1], inplace=True)

clf = top.PHNovDet(max_dimension=1, threshold=threshold, base=base_lower, ratio=0.85625, M=3, random_state=28,
                   shuffle=False, sparse=3.0)
clf.fit(x_train)

predicted = clf.predict(x_test)
print(predicted)

y_scores = clf.score_samples(x_test)
print(y_scores)

roc.plot(y_test=np.array(y_test), y_scores=y_scores, pos_label=-1, title='PH - ')
