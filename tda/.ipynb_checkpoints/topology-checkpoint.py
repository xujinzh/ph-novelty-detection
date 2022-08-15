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
from tda.flatten import flatten
import multiprocessing
from multiprocessing import Pool
import os
from utils.now import current_time

torch.multiprocessing.set_start_method('forkserver', force=True)
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

    .. version added:: 3.0.0

    :parameter
    max_edge_length : int, optional (default=42)
        The max edge length to construct simplicial complex for TDA

    max_dimension : int, optional (default=1)
        The max dimension of simplicial complex (Rips complex)

    homology_coefficient_filed : int, optional (default=2)
        The field for construct homology group

    min_persistence : int, optional (default=0)

    sparse : float, optional (default=1.0)
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

    e : float, optional (default=0.0)
        If `e` is 0, this uses an expensive algorithm to compute the exact distance.
        If `e` is not 0, it asks for an additive `e`-approximation, and currently also allows a small multiplicative error (the last 2 or 3 bits of the mantissa may be wrong). This version of the algorithm takes advantage of the limited precision of `double` and is usually a lot faster to compute, whatever the value of `e`.
        Thus, by default (`e=None`), `e` is the smallest positive double.
        
    use_gpu : str, optional (default='yes')
        use GPU or not
        
    max_clusters : int, optional (default=300)
        maximum number of clusters allowed
        
    mp_score : boolean, optional (default=Trie)
        calculate prediction scores using multiprocessing
        
    References
    ----------
    .. [1] Jinzhong Xu, Junrong Du, Ye Li, Lele Xu, Lili Guo, Xuzhi Li.
        Novelty Detection with Topological Signatures.
    """

    scores = []

    def __init__(self, max_edge_length=12.0, max_dimension=1, homology_coefficient_field=2,
                 min_persistence=0, sparse=1.0, threshold=0.5, base=15, ratio=0.25,
                 standard_deviation_coefficient=3, random_state=42, shuffle=True, cross_separation=3,
                 e=0.0, use_gpu='yes', max_clusters=300, mp_score=True):
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
        self.max_clusters = max_clusters
        self.mp_score = mp_score  # 训练时使用单进程计算测试分数。在外层使用多进程

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
            # 使用 CPU 进行计算持续条形码
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
    def representative_points(x_data, labels):
        """
        从数据x_data和其聚类标签计算得到代表点，用于近似数据拓扑
        """
        unique_labels = Counter(labels)
        for i, label in enumerate(unique_labels):
            # 首先对于每一个聚类标签，计算各分位点和均值
            qua_big = np.quantile(x_data[labels == label], .75, axis=0)
            qua_small = np.quantile(x_data[labels == label], .25, axis=0)
            median = np.median(x_data[labels == label], axis=0)
            mean = np.mean(x_data[labels == label], axis=0)
            # 其次对于第一个标签，把分位点组合成代表点
            if i == 0:
                if unique_labels[i] <= len(x_data) // len(unique_labels):
                    shape_data = np.vstack((median, mean))
                else:
                    shape_data = np.vstack((qua_big, qua_small, median, mean))
            # 最后对于其他标签，依次累加分位点到代表点，组成更大的代表点
            else:
                if unique_labels[i] <= len(x_data) // len(unique_labels):
                    shape_data = np.vstack((median, mean, shape_data))
                else:
                    shape_data = np.vstack((qua_big, qua_small, median, mean, shape_data))
        return shape_data

    @property
    def fit(self):
        return self._fit

    def _fit(self, x_data=None, y_data=None, cluster='kmeans', n_cluster=20, branching_factor=100,
             cluster_threshold=1.0, eps=3, min_samples=3, linkage='ward'):

        x_data = check_array(x_data)
        x_data = np.array(x_data)

        print(f"{current_time()} 开始使用聚类算法{cluster}进行聚类")

        start_time = time()
        if cluster == 'tomato': # n_jobs=-1 会报错
            model = Tomato(density_type="DTM", n_clusters=n_cluster)
        elif cluster == 'birch':
            model = Birch(n_clusters=n_cluster, branching_factor=branching_factor,
                          threshold=cluster_threshold)
        elif cluster == 'dbscan':
            model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=3)
        elif cluster == 'optics':
            model = OPTICS(eps=eps, min_samples=min_samples, algorithm='auto', n_jobs=3)
        elif cluster == 'hierarchical':
            model = AgglomerativeClustering(n_clusters=n_cluster, linkage=linkage)
        elif cluster == 'spectral':
            model = SpectralClustering(n_clusters=n_cluster, assign_labels="discretize",
                                       eigen_solver='arpack', affinity="nearest_neighbors",
                                       random_state=self.random_state)
        else:
            model = KMeans(n_clusters=n_cluster, random_state=self.random_state).fit(x_data)
        labels = model.fit_predict(x_data)
        end_time = time()
        print(f"{current_time()} 聚类消耗时间：", (end_time - start_time))
        print(f"{current_time()} 聚类中心的个数是：", len(set(labels)))

        start_time = time()
        if len(set(labels)) == 1:
            # # 如果只有一个聚类，则先把数据划分为多个小数据集，分别聚类再求代表点，合并后作为整个数据集的代表点
            # print("划分成小数据集后再聚类")
            # x_data_small = np.array_split(x_data, 20)
            # for flag, small_data in enumerate(x_data_small):
            #     small_labels = model.fit_predict(small_data)
            #     print(f"第{flag}个小数据集聚类得到类别数{set(small_labels)}")
            #     small_shape_data = self.representative_points(x_data=small_data, labels=small_labels)
            #     if flag == 0:
            #         self.shape_data = small_shape_data
            #     else:
            #         self.shape_data = np.vstack((small_shape_data, self.shape_data))
            # # 上面划分为小数据集方法效果不好，这里采用随机选取代表点
            # print("random choice points for shape data")
            # self.shape_data = x_data[np.random.choice(x_data.shape[0], 20, replace=False)]

            # 针对只有一个聚类的情况，从聚类中多取一些代表点，如，mean, quantile
            print("聚类个数是1，选择100个分位点作为代表点")
            self.shape_data = np.mean(x_data, axis=0)
            for i in np.arange(0, 1.01, 0.01):
                points = np.quantile(x_data, i, axis=0)
                self.shape_data = np.vstack([self.shape_data, points])
        elif len(set(labels)) >= self.max_clusters:
            print(f"聚类个数大于 {self.max_clusters}, 所以进行摊平")
            labels = flatten(labels=labels, centroid=self.max_clusters)
            self.shape_data = self.representative_points(x_data=x_data, labels=labels)
        else:
            self.shape_data = self.representative_points(x_data=x_data, labels=labels)
        self.shape = self._diagram(self.shape_data)
        end_time = time()
        print(f"{current_time()} 计算数据形状消耗：", (end_time - start_time))
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

    def score(self, label_data):
        label = label_data[0]
        point = label_data[1]
        data_novelty = np.vstack((self.shape_data, point))
        diagram_novelty = self._diagram(data_novelty)
        score_ = np.min([self._bottleneck(self.shape, diagram_novelty), 99])
        return label, score_

    def _score_samples(self, x_test):
        # 多进程计算测试集的分数
        if self.mp_score:
            ds = [(i, d) for i, d in enumerate(x_test)]
            p = Pool(processes=min(os.cpu_count() - 3, 5))
            res = p.map(self.score, ds)
            res = sorted(res, key=lambda x: x[0])
            self.scores = [x[1] for x in res]
        # 单进程计算测试集的分数
        else:
            self.scores = []
            for i in range(len(x_test)):
                point = x_test[i]
                data_novelty = np.vstack((self.shape_data, point))
                diagram_novelty = self._diagram(data_novelty)
                self.scores.append(np.min([self._bottleneck(self.shape, diagram_novelty), 99]))
        return np.array(self.scores)
