import numpy as np
import pandas as pd
from tda import topology as top
from tda import roc
import copy

novelty = 10
normal = novelty * 2
test_num = novelty + normal  # 10 novelty point and 20 normal point
head_train = test_num + 10  # the index of first novelty point
span = 100  # the number of train set, and the ratio * span is the base shape data set
tail_train = head_train + span
base_lower = 10  # the minimum of the points in base shape data set
threshold = 1.0

input_path = "./data/breast-cancer-unsupervised-ad.csv"
breast = pd.read_csv(input_path, header=None)
x_train = np.array(breast.iloc[head_train:tail_train, :-1])
x_test = np.array(breast.iloc[:test_num, :-1])
y_test = copy.deepcopy(breast.iloc[:test_num, -1])
y_test.replace(['o', 'n'], [-1, 1], inplace=True)

clf = top.PHNovDet(max_dimension=2, threshold=threshold, base=base_lower, ratio=0.125, M=3, random_state=28,
                   shuffle=False, sparse=0, max_edge_length=6)
clf.fit(x_train)

predicted = clf.predict(x_test)
print(predicted)

y_scores = clf.score_samples(x_test)
print(y_scores)

roc.area(y_test=np.array(y_test), y_scores=y_scores, pos_label=-1, title='PH - ')
