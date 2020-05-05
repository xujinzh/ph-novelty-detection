import numpy as np
import pandas as pd
from tda import topology as top
from tda import roc
import copy

novelty = 75
normal = novelty * 2
test_num = novelty + normal
head_train = test_num + 10
span = 100
tail_train = head_train + span
base_lower = 20
threshold = 1.0

input_path = "./data/satellite-unsupervised-ad.csv"
satellite = pd.read_csv(input_path, header=None)
x_train = np.array(satellite.iloc[head_train:tail_train, :-1])
x_test = np.array(satellite.iloc[:test_num, :-1])
y_test = copy.deepcopy(satellite.iloc[:test_num, -1])
y_test.replace(['o', 'n'], [-1, 1], inplace=True)

clf = top.PHNovDet(max_dimension=2, threshold=threshold, base=base_lower, ratio=0.85625, M=3, random_state=28,
                   shuffle=False, sparse=0)
clf.fit(x_train)

predicted = clf.predict(x_test)
print(predicted)

y_scores = clf.score_samples(x_test)
print(y_scores)

roc.area(y_test=np.array(y_test), y_scores=y_scores, pos_label=-1, title='PH - ')
