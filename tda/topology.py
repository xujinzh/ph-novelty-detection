# Authors: Jinzhong Xu <jinzhongxu@csu.ac.cn>
#          Xuzhi Li <xzhli@csu.ac.cn>
# License: BSD 3-Clause "New" or "Revised" License

import gudhi
import numpy as np
import pandas as pd
from sklearn import preprocessing
from tqdm import tqdm
import copy
from tda.timestamps import displaytime
from sklearn.model_selection import train_test_split
import warnings

__all__ = ["PHNovDet"]

__author__ = "Jinzhong Xu and Xuzhi Li"
__version__ = "2.0.0"
__license__ = 'BSD 3-Clause "New" or "Revised" License'

__available_modules = ''
__missing_modules = ''


class PHNovDet(object):
    """Unsupervised Novelty Detection using Persistent Homology (PH)

    The persistent homology is a method of topological data analysis,
    which study the shape of data set. It measures the difference
    between the shape of data set and data set with a sample point.
    It is global in that the persistent diagram represent
    the topology features of data set. More precisely, bottleneck distance
    is used to estimate the global shape changes. By comparing
    the bottleneck distance to a threshold, one can identify samples
    that have a bigger influence to the shape of the data set.
    These are considered novelties.

    .. versionadded:: 2.0.0

    :parameter
    max_edge_length : int, optional (default=42)
        The max edge length to construct simplicial complex for TDA

    max_dimension : int, optional (default=1)
        The max dimension of simplicial complex (Rips complex)

    homology_coeff_filed : int, optional (default=2)
        The field for construct homology group

    min_persistence : int, optional (default=0)

    sparse : float, optional (default=0)
        For decrease simplex in filtration and short calculation time,
        but may be decrease the accuracy for Rips complex.

    threshold : float, optional (default=0.05)
        Judge novelty by the threshold

    base : int, optional (default=20)
        The cardinality of base shape data set

    ratio : float, optional (default=0.2)
        The ratio control the base shape data set which is equal to ratio times x_train

    M : float, optional (default=3)
        The multiplier that control the threshold (mu + M * sigma)

    random_state : int, optional (default=26)
        train test split argument

    shuffle : boolean, optional (default=True)
        train test split argument

    References
    ----------
    .. [1] Jinzhong Xu, Xuzhi Li, Ye Li, Lele Xu, Lili Guo, Junrong Du.
        Novelty Detection with Topological Signatures.
    """

    def __init__(self, data=None, max_edge_length=12.0, max_dimension=1, homology_coeff_field=2, min_persistence=0,
                 sparse=0.0, threshold=0.5, base=20, ratio=0.5, M=3, random_state=42, shuffle=True):
        self.data = data
        self.max_edge_length = max_edge_length
        self.max_dimension = max_dimension
        self.homology_coeff_field = homology_coeff_field
        self.sparse = sparse
        self.min_persistence = min_persistence
        self.threshold = threshold
        self.base = base
        self.scores = np.array([])
        self.ratio = ratio
        self.M = M
        self.random_state = random_state
        self.shuffle = shuffle
        self.sparse = sparse

    def _ph(self, plot=False):
        data = preprocessing.scale(self.data)
        # scaled data has zero mean (\mu = 0) and unit variance (\sigma = 0)
        rips_complex = gudhi.RipsComplex(points=data, max_edge_length=self.max_edge_length, sparse=self.sparse)
        # sparse for save time
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=self.max_dimension)
        diagram = simplex_tree.persistence(homology_coeff_field=self.homology_coeff_field,
                                           min_persistence=self.min_persistence)
        if plot:
            ph_plot = gudhi.plot_persistence_diagram(diagram)
            ph_plot.show()

        return diagram

    def _bottleneck(self, point, plot=False):
        _diagram = self._ph(plot=plot)
        diagram = []
        for i, d in enumerate(_diagram):
            diagram.append(list(_diagram[i][1]))

        temp_data = self.data
        self.data = np.vstack((self.data, point))
        _diagram_novelty = self._ph(plot=plot)
        diagram_novelty = []
        for i, d in enumerate(_diagram_novelty):
            diagram_novelty.append(list(_diagram_novelty[i][1]))

        self.data = temp_data

        return gudhi.bottleneck_distance(diagram, diagram_novelty, 0.05)  # e = 0.05 for save time

    def fit(self, X, y=None):
        X_train, X_test, y_train, y_test = train_test_split(X, range(len(X)), test_size=self.ratio,
                                                            random_state=self.random_state, shuffle=self.shuffle)
        self.data = X_train  # Base shape data set
        thresholds = []

        try:  # for tqdm print time consume
            with tqdm(range(len(X_test))) as t:
                for i in t:
                    thresholds.append(np.min([self._bottleneck(point=X_test[i]), 9.9]))
                self.threshold = np.mean(thresholds) + self.M * np.std(thresholds, ddof=1)  # T
                # ddof = 1 represent unbiased sample standard deviation
        except KeyboardInterrupt:
            t.close()
            raise
        t.close()

        return self.threshold

    def predict(self, x):  # compute -1 and 1 respectively for outlier point and inlier point
        y_scores = []
        for i in range(len(x)):
            point = x[i]
            y_scores.append(np.min([self._bottleneck(point=point), 99]))
        y_scores = np.array(y_scores)
        self.scores = copy.deepcopy(y_scores)  # backup scores value for function score_samples()
        max_score = np.max(y_scores) + 1
        scores = y_scores / max_score

        T = self.threshold / max_score
        scores[scores >= T] = 1  # big distance convert to 1
        scores[scores < T] = -1  # small distance convert to -1
        y_scores = scores.astype(int)
        return -y_scores  # outlier point label -1 and inlier point label 1

    # @displaytime
    def score_samples(self, x):
        self.predict(x)
        return self.scores

    def OX_fit(self, X, output_path='.'):
        '''
        param x_train: the train set
        param output_path: save the base shape dataset B to output path
        '''
        self.data = X
        data = pd.DataFrame(data=X)
        base_list = list(data.index)
        abandom_point = []

        if self.base > len(data):
            print("the base shape data set is too small")
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
                            if self._bottleneck(choice_point) < self.threshold:
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
