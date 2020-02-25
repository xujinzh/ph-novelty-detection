import numpy as np
import pandas as pd
import Topology as top

input_path = "../../notebook/ph-novelty/data/satellite-unsupervised-ad.csv"
satellite = pd.read_csv(input_path, header=None)
outlier_number = 300
base_lower = 25
threshold = 0.618
output_path = "/./base" + str(base_lower) + "-thr" + str(threshold)

X_train = np.array(satellite.iloc[outlier_number:, :-1])
model = top.PHNovDet(max_dimension=1, threshold=threshold, base=base_lower)
model.fit(X_train, output_path=output_path)
X_test = np.array(satellite.iloc[:outlier_number, :-1])
predicted = model.predict(X_test)
print(predicted)
scores = model.score_samples()
print(scores)
