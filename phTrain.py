import numpy as np
import pandas as pd
from tda import topology as top
from tda import roc

input_path = "./data/satellite-unsupervised-ad.csv"
satellite = pd.read_csv(input_path, header=None)
outlier_number = 150
span = 100
base_lower = 20
threshold = 0.35

x_train = np.array(satellite.iloc[outlier_number:outlier_number + span, :-1])
model = top.PHNovDet(max_dimension=1, threshold=threshold, base=base_lower, ratio=0.80, M=2.5, random_state=26)

# output_path = "./output/base" + str(base_lower) + "-thr" + str(threshold) + "-out" + str(outlier_number)
# model.fit(x_train=x_train, output_path=output_path)

T = model.fit(data=x_train)
print("threshold: ", T)

x_test = np.array(satellite.iloc[:150, :-1])

predicted = model.predict(x_test=x_test)
print(predicted)

y_scores = model.score_samples()
print(y_scores)

y_test = [1] * 75 + [0] * 75
roc.plot(y_test=np.array(y_test), y_scores=y_scores)
