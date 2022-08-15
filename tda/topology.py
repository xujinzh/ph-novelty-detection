# Authors: Jinzhong Xu <jinzhongxu@csu.ac.cn>
# License: BSD 3-Clause "New" or "Revised" License

import math
import sys
from collections import Counter
from time import time

import gudhi
import numpy as np
import ripserplusplus as rpp_py
import torch
from gudhi.clustering.tomato import Tomato
from sklearn import preprocessing
from sklearn.cluster import Birch, KMeans, DBSCAN, OPTICS, AgglomerativeClustering, SpectralClustering
from sklearn.utils.validation import check_array

sys.setrecursionlimit(999999)

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

    def __init__(self, max_edge_length=12.0, max_dimension=1, homology_coefficient_field=2,
                 min_persistence=0, sparse=1.0, threshold=0.5, base=15, ratio=0.25,
                 standard_deviation_coefficient=3, random_state=42, shuffle=True, cross_separation=3,
                 e=0.0, use_gpu='yes'):
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
        self.shape = None
        self.shape_data = None
        self.cross_separation = cross_separation
        self.e = e
        self.use_gpu = use_gpu

    @property
    def ph(self):
        return self._ph

    def _ph(self, points):
        """
        compute persistent diagram by geometry understanding in higher dimensions package
        :param points: point cloud
        :return: persistent diagram
        """
        if (torch.cuda.is_available()) and (self.use_gpu == 'yes'):
            # 使用 GPU 加速计算持续条形码
            d = rpp_py.run("--format point-cloud", preprocessing.minmax_scale(points))
            diagram = [(k, v.item()) for k in d.keys() for v in d[k]]
        else:
            # 只使用 CPU 进行计算持续条形码
            points = preprocessing.minmax_scale(points)
            rips_complex = gudhi.RipsComplex(points=points, sparse=self.sparse)
            simplex_tree = rips_complex.create_simplex_tree(max_dimension=self.max_dimension)
            diagram = simplex_tree.persistence(homology_coeff_field=self.homology_coefficient_field,
                                               min_persistence=self.min_persistence)
        return diagram

    @property
    def diagram(self):
        return self._diagram

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

    @staticmethod
    def small_into_big(labels, small_value, big_value):
        """
        把一个列表labels中指定的元素small_value，用一个值big_value替换
        """
        labels = [big_value if value == small_value else value for value in labels]
        return labels

    def flatten(self, labels, centroid=20):
        """
        把列表中的元素按照出现的次数摊平，使得元素出现次数趋近均等
        具体地，将出现最小的元素用第二小的元素替换，如果两者都小于平均个数
        """
        # 统计每个元素出现的次数
        counter_labels = Counter(labels)
        # 如果聚类中心数小于 centroid，则不摊平
        if len(counter_labels) <= centroid:
            return labels
        # 按照出现的次数从大到小排序
        sort_labels = counter_labels.most_common()
        # 平均元素个数阈值
        thres = math.ceil(len(labels) / len(sort_labels))
        # 如果最小次数的元素和第二下次数的元素都出现次数都小于均值，
        # 那么把最小元素用第二下元素替换
        # 递归执行检查和替换
        if (sort_labels[-1][1] < thres) and (sort_labels[-2][1] < thres):
            labels = self.small_into_big(labels=labels, small_value=sort_labels[-1][0],
                                         big_value=sort_labels[-2][0])
            return self.flatten(labels)
        else:
            return labels

    @property
    def fit(self):
        return self._fit

    def _fit(self, x_data=None, y_data=None, cluster='kmeans', n_cluster=20, branching_factor=100,
             cluster_threshold=1.0, eps=3, min_samples=3, linkage='ward'):

        x_data = check_array(x_data)
        x_data = np.array(x_data)

        print(f"开始使用聚类算法{cluster}进行聚类")

        start_time = time()
        if cluster == 'tomato':
            model = Tomato(density_type="DTM", n_clusters=n_cluster, n_jobs=-1)
        elif cluster == 'birch':
            model = Birch(n_clusters=n_cluster, branching_factor=branching_factor,
                          threshold=cluster_threshold)
        elif cluster == 'dbscan':
            model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        elif cluster == 'optics':
            model = OPTICS(eps=eps, min_samples=min_samples, n_jobs=-1)
        elif cluster == 'hierarchical':
            model = AgglomerativeClustering(n_clusters=n_cluster, linkage=linkage)
        elif cluster == 'spectral':
            model = SpectralClustering(n_clusters=n_cluster, assign_labels="discretize",
                                       eigen_solver='arpack', affinity="nearest_neighbors",
                                       random_state=self.random_state, n_jobs=-1)
        else:
            model = KMeans(n_clusters=n_cluster, random_state=self.random_state).fit(x_data)

        labels = model.fit_predict(x_data)
        end_time = time()
        print("聚类消耗时间：", (end_time - start_time))

        print("聚类中心的个数是：", len(set(labels)))
        # print('摊平前 聚类中心有：', len(set(labels)))
        # 经测试，摊平后检测效果不好，因此，不建议摊平
        # labels = self.flatten(labels=labels, centroid=20)
        # print('摊平后 聚类中心有：', len(set(labels)))
        # set导致精度降低
        # unique_labels = list(set(labels))
        unique_labels = Counter(labels)

        start_time = time()
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
        end_time = time()
        print("计算数据形状消耗：", (end_time - start_time))
        return self.shape_data, self.shape

    @property
    def predict(self):
        return self._predict

    def _predict(self, x_test):  # compute -1 and 1 respectively for outlier point and internal point
        """
        compute -1 and 1 respectively for novelty sample and normal sample
        :param x_test: the data set for predict
        :return: the predicted label (-1 and 1)
        """
        x_test = check_array(x_test)
        x_test = np.array(x_test)

        # binary = preprocessing.Binarizer(threshold=self.threshold)  # convert to 0 or 1
        # scores = binary.transform([self._score_samples(x_test)])
        # is_inlier = [1 - 2 * x for x in scores.tolist()[0]]  # convert 0 or 1 to 1 or -1
        # is_inlier = [int(i) for i in is_inlier]

        scores = self._score_samples(x_test)
        is_inlier = np.ones(len(scores), dtype=int)
        is_inlier[scores > self.threshold] = -1

        return is_inlier  # Returns -1 for anomalies/outliers and +1 for inliers

    @property
    def score_samples(self):
        return self._score_samples

    def _score_samples(self, x_test):
        self.scores = []
        for i in range(len(x_test)):
            point = x_test[i]
            data_novelty = np.vstack((self.shape_data, point))
            diagram_novelty = self._diagram(data_novelty)
            self.scores.append(np.min([self._bottleneck(self.shape, diagram_novelty), 99]))
        return np.array(self.scores)
