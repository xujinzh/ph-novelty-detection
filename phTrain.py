import numpy as np
import pandas as pd
from tda import topology as top

input_path = "./data/satellite-unsupervised-ad.csv"
satellite = pd.read_csv(input_path, header=None)
outlier_number = 3000
base_lower = 25
threshold = 0.618
output_path = "./output/base" + str(base_lower) + "-thr" + str(threshold) + "-out" + str(outlier_number)

X_train = np.array(satellite.iloc[outlier_number:, :-1])
model = top.PHNovDet(max_dimension=1, threshold=threshold, base=base_lower)
model.fit(X_train, output_path=output_path)
X_test = np.array(satellite.iloc[:150, :-1])
predicted = model.predict(X_test)
print(predicted)
scores = model.score_samples()
print(scores)
