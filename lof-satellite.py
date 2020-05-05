from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import pandas as pd
from tda import roc
import copy

novelty = 75
normal = novelty * 2
test_num = novelty + normal
head_train = test_num + 10
span = 100
tail_train = head_train + span

input_path = "./data/satellite-unsupervised-ad.csv"
satellite = pd.read_csv(input_path, header=None)
x_train = np.array(satellite.iloc[head_train:tail_train, :-1])
x_test = np.array(satellite.iloc[:test_num, :-1])
y_test = copy.deepcopy(satellite.iloc[:test_num, -1])
y_test.replace(['o', 'n'], [-1, 1], inplace=True)

clf = LocalOutlierFactor(novelty=True, n_neighbors=9)

T = clf.fit(x_train)
print("threshold: ", T)

predicted = clf.predict(x_test)
print(predicted)

y_scores = clf.score_samples(x_test)
print(y_scores)

roc.area(y_test=np.array(y_test), y_scores=y_scores, pos_label=1, title='LOF - ')

