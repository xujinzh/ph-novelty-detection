# Authors: Jinzhong Xu <jinzhongxu@csu.ac.cn>
# License: BSD 3-Clause "New" or "Revised" License

import gudhi
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import Birch, KMeans, DBSCAN, OPTICS, AgglomerativeClustering, SpectralClustering
from gudhi.clustering.tomato import Tomato
from collections import Counter

__all__ = ["PHNovDet"]
__author__ = "Jinzhong Xu"
__version__ = "3.0.0"
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

    .. version added:: 2.0.0

    :parameter
    max_edge_length : int, optional (default=42)
        The max edge length to construct simplicial complex for TDA

    max_dimension : int, optional (default=1)
        The max dimension of simplicial complex (Rips complex)

    homology_coefficient_filed : int, optional (default=2)
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
    .. [1] Jinzhong Xu, Junrong Du, Ye Li, Lele Xu, Lili Guo, Xuzhi Li.
        Novelty Detection with Topological Signatures.
    """

    scores = []

    def __init__(self, max_edge_length=12.0, max_dimension=1, homology_coefficient_field=2, min_persistence=0,
                 sparse=1.0, threshold=0.5, base=15, ratio=0.25, standard_deviation_coefficient=3, random_state=42,
                 shuffle=True, cross_separation=3, e=0.0):
        self.max_edge_length = max_edge_length
        self.max_dimension = max_dimension
        self.homology_coefficient_field = homology_coefficient_field
        self.sparse = sparse
        self.min_persistence = min_persistence
        self.threshold = threshold
        self.base = base
        self.ratio = ratio
        self.standard_deviation_coefficient = standard_deviation_coefficient
        self.random_state = random_state
        self.shuffle = shuffle
        self.sparse = sparse
        self.shape = None
        self.shape_data = None
        self.cross_separation = cross_separation
        self.e = e

    def _ph(self, points):
        """
        compute persistent diagram by geometry understanding in higher dimensions package
        :param points: point cloud
        :return: persistent diagram
        """
        points = preprocessing.minmax_scale(points)
        rips_complex = gudhi.RipsComplex(points=points, sparse=self.sparse)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=self.max_dimension)
        diagram = simplex_tree.persistence(homology_coeff_field=self.homology_coefficient_field,
                                           min_persistence=self.min_persistence)
        return diagram

    def _diagram(self, points):
        """
        return diagram in list form
        :param points: point cloud
        :return: persistent diagram in list form
        """
        return [p[1] for p in self._ph(points)]

    def _bottleneck(self, diag1, diag2):
        """
        compute bottleneck distance between diagram and diagram_novelty
        :param diag1: persistent diagram for shape data in list form
        :param diag2: persistent diagram for cup dataset with novelty data and shape date in list form
        :return: bottleneck distance
        """
        return gudhi.bottleneck_distance(diag1, diag2, self.e)

    def fit(self, x_data=None, y_data=None, cluster='kmeans', n_cluster=20, branching_factor=100,
            threshold=1.0, eps=3, min_samples=3, linkage='ward'):

        if cluster == 'tomato':
            model = Tomato(density_type="DTM", n_clusters=n_cluster, n_jobs=-1)
        elif cluster == 'birch':
            model = Birch(n_clusters=n_cluster, branching_factor=branching_factor, threshold=threshold)
        elif cluster == 'dbscan':
            model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        elif cluster == 'optics':
            model = OPTICS(eps=eps, min_samples=min_samples, n_jobs=-1)
        elif cluster == 'hierarchical':
            model = AgglomerativeClustering(n_clusters=n_cluster, linkage=linkage)
        elif cluster == 'spectral':
            model = SpectralClustering(n_clusters=n_cluster, assign_labels="discretize", eigen_solver='arpack',
                                       affinity="nearest_neighbors", random_state=self.random_state,
                                       n_jobs=-1)
        else:
            model = KMeans(n_clusters=n_cluster, random_state=self.random_state).fit(x_data)

        labels = model.fit_predict(x_data)
        unique_labels = Counter(labels)

        if len(unique_labels) < 2:
            return 0

        for i, label in enumerate(unique_labels):
            qua_big = np.quantile(x_data[labels == label], .75, axis=0)
            qua_small = np.quantile(x_data[labels == label], .25, axis=0)
            median = np.median(x_data[labels == label], axis=0)
            mean = np.mean(x_data[labels == label], axis=0)
            if i == 0:
                if unique_labels[i] <= len(x_data) // len(unique_labels):
                    self.shape_data = np.vstack((median, mean))
                else:
                    self.shape_data = np.vstack((qua_big, qua_small, median, mean))
            else:
                if unique_labels[i] <= len(x_data) // len(unique_labels):
                    self.shape_data = np.vstack((median, mean, self.shape_data))
                else:
                    self.shape_data = np.vstack((qua_big, qua_small, median, mean, self.shape_data))

        self.shape = self._diagram(self.shape_data)
        return self.shape_data, self.shape

    def predict(self, x_test):  # compute -1 and 1 respectively for outlier point and internal point
        """
        compute -1 and 1 respectively for novelty sample and normal sample
        :param x_test: the data set for predict
        :return: the predicted label (-1 and 1)
        """
        self.scores = []
        for i in range(len(x_test)):
            point = x_test[i]
            data_novelty = np.vstack((self.shape_data, point))
            diagram_novelty = self._diagram(data_novelty)
            self.scores.append(np.min([self._bottleneck(self.shape, diagram_novelty), 99]))
        scores = np.array(self.scores)
        binary = preprocessing.Binarizer(threshold=self.threshold)  # convert to 0 or 1
        scores = binary.transform([scores])
        predict = [1 - 2 * x for x in scores.tolist()[0]]  # convert 0 or 1 to 1 or -1
        predict = [int(i) for i in predict]
        return predict  # outlier point label -1 and internal point label 1

    def score_samples(self, x_test):
        self.predict(x_test)
        return self.scores
