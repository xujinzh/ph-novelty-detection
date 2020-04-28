import gudhi as gh
import numpy as np
import pandas as pd
from sklearn import preprocessing
from tqdm import tqdm
import copy
from tda.timestamps import displaytime
from sklearn.model_selection import train_test_split


class PHNovDet(object):

    def __init__(self, data=None, max_edge_length=42, max_dimension=1, homology_coeff_field=2, min_persistence=0,
                 threshold=0.05, base=20, ratio=0.2, M=3, random_state=26, shuffle=True):
        self.data = data
        self.max_edge_length = max_edge_length
        self.max_dimension = max_dimension
        self.homology_coeff_field = homology_coeff_field
        self.min_persistence = min_persistence
        self.threshold = threshold
        self.base = base
        self.scores = np.array([])
        self.ratio = ratio
        self.M = M
        self.random_state = random_state
        self.shuffle = shuffle

    def ph(self, plot=True):

        #         print("RipsComplex creation from points")
        data = preprocessing.scale(self.data)
        rips = gh.RipsComplex(points=data, max_edge_length=self.max_edge_length)
        simplex_tree = rips.create_simplex_tree(max_dimension=self.max_dimension)
        diag = simplex_tree.persistence(homology_coeff_field=self.homology_coeff_field,
                                        min_persistence=self.min_persistence)
        #         print("diag=", diag)

        if plot == True:
            pplot = gh.plot_persistence_diagram(diag)
            pplot.show()
        return diag

    def bottleneck(self, point, plot=True):
        diag = self.ph(plot)
        diag0 = []
        for i, d in enumerate(diag):
            diag0.append(list(diag[i][1]))

        temp_data = self.data
        self.data = np.vstack((self.data, point))
        diag_point = self.ph(plot)
        diag_point0 = []
        for i, d in enumerate(diag_point):
            diag_point0.append(list(diag_point[i][1]))

        self.data = temp_data

        return gh.bottleneck_distance(diag0, diag_point0)

    # @displaytime
    def fit(self, x_train, output_path='.'):
        '''
        param x_train: the train set
        param output_path: save the base shape dataset B to output path
        '''
        self.data = x_train
        data = pd.DataFrame(data=x_train)
        base_list = list(data.index)
        abandom_point = []

        if self.base > len(data):
            print("基础数据集的势太小")
        else:
            try:  # for tqdm print time consume
                with tqdm(np.arange(len(data.index))) as t:
                    for i in t:
                        choice_index = i
                        choice_point = np.array(data.iloc[choice_index, :])
                        temp_list = copy.deepcopy(base_list)
                        base_list.remove(choice_index)
                        self.data = np.array(data.iloc[base_list, :])
                        if len(self.data) < self.base:
                            break
                        else:
                            if self.bottleneck(choice_point, plot=False) < self.threshold:
                                base_list = copy.deepcopy(temp_list)
                            else:
                                abandom_point.append(choice_index)
                                continue
            except KeyboardInterrupt:
                t.close()
                raise

            t.close()

            B = data.iloc[base_list, :]
            self.data = np.array(B)  # the model is being fitted by train set
            B.to_csv(output_path + ".csv")
            print("self data shape: ", data.shape[0])
            print("B data shape: ", B.shape)
            with open(output_path + ".txt", 'w') as f:  # write abandom point to file
                for item in abandom_point:
                    f.write("%s\n" % item)

    # @displaytime
    def fit(self, data):
        X_train, X_test, y_train, y_test = train_test_split(data, range(len(data)), test_size=self.ratio,
                                                            random_state=self.random_state, shuffle=self.shuffle)
        self.data = X_train  # B
        onepoidis = []

        try:  # for tqdm print time consume
            with tqdm(range(len(X_test))) as t:
                for i in t:
                    onepoidis.append(self.bottleneck(X_test[i], plot=False))
                self.threshold = np.mean(onepoidis) + self.M * np.std(onepoidis, ddof=1)  # T
                # ddof = 1 represent unbiased sample standard deviation
        except KeyboardInterrupt:
            t.close()
            raise
        t.close()

        return self.threshold

    # @displaytime
    def predict(self, x_test):  # compute -1 and 1 respectively for outlier point and inlier point
        y_scores = []
        for i in range(len(x_test)):
            point = x_test[i]
            y_scores.append(self.bottleneck(point, plot=False))
        y_scores = np.array(y_scores)
        self.scores = copy.deepcopy(y_scores)  # backup scores value for function score_samples()
        R = np.max(y_scores) + 1
        scores = y_scores/R

        T = self.threshold/R
        scores[scores >= T] = 1  # big distance convert to 1
        scores[scores < T] = -1 # small distance convert to -1
        y_scores = scores.astype(int)
        return -y_scores  # outlier point label -1 and inlier point label 1

    # @displaytime
    def score_samples(self, x_test):
        self.predict(x_test)
        return self.scores
