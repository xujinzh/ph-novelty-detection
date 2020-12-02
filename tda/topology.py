# Authors: Jinzhong Xu <jinzhongxu@csu.ac.cn>
# License: BSD 3-Clause "New" or "Revised" License

import gudhi
import numpy as np
import pandas as pd
from sklearn import preprocessing
from tqdm import tqdm
import copy
from tda.timestamps import display_time
from sklearn.model_selection import train_test_split
import warnings
import collections
import operator
import random
import os
from sklearn.cluster import Birch, KMeans, DBSCAN, OPTICS, AgglomerativeClustering, SpectralClustering
from gudhi.clustering.tomato import Tomato

__all__ = ["PHNovDet"]
__author__ = "Jinzhong Xu"
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
                 sparse=0.0, threshold=0.5, base=15, ratio=0.25, standard_deviation_coefficient=3, random_state=42,
                 shuffle=True, cross_separation=3):
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

    def _diagram(self, points):
        """
        return diagram in list form
        :param points: point cloud
        :return: persistent diagram in list form
        """
        return [p[1] for p in self._ph(points)]

    def _ph(self, points):
        """
        compute persistent diagram by geometry understanding in higher dimensions package
        :param points: point cloud
        :return: persistent diagram
        """
        points = preprocessing.minmax_scale(points)
        rips_complex = gudhi.RipsComplex(points=points)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)
        return simplex_tree.persistence()

        # points = preprocessing.minmax_scale(points)
        # # minmax_scale(normalization) need less time than scale(standardization)
        # # because max_edge_length is ceil(sqrt{numbers of variables}) in minmax_scale; but that is bigger in scale
        #
        # rips_complex = gudhi.RipsComplex(points=points, max_edge_length=self.max_edge_length, sparse=self.sparse)
        # # self.sparse for save time but have different result in every time run
        # simplex_tree = rips_complex.create_simplex_tree(max_dimension=self.max_dimension)
        # diagram = simplex_tree.persistence(homology_coeff_field=self.homology_coefficient_field,
        #                                    min_persistence=self.min_persistence)
        # return diagram

    @staticmethod
    def _bottleneck(diag1, diag2):
        """
        compute bottleneck distance between diagram and diagram_novelty
        :param diag1: persistent diagram for shape data in list form
        :param diag2: persistent diagram for cup dataset with novelty data and shape date in list form
        :return: bottleneck distance
        """
        return gudhi.bottleneck_distance(diag1, diag2)
        # return gudhi.bottleneck.PyCapsule.bottleneck_distance(diagram, diagram_novelty, 0.05)  # e = 0.05 节省时间

    def cross_fit(self, x_data=None, y_data=None):
        # 交叉划分数据集为训练集和验证集，用训练集作为基数据测试验证集里每一个样本点，选择前1/3
        for i in range(self.cross_separation):
            self.random_state += 3
            x_train, x_cv, y_train, y_cv = train_test_split(x_data, y_data, train_size=self.ratio,
                                                            random_state=self.random_state)
            bd = {}
            for j in range(len(x_cv)):
                train_one_cv = np.vstack((x_train, x_cv[j]))
                bd[j] = self._bottleneck(self._diagram(x_train), self._diagram(train_one_cv))
            sorted_bd = collections.OrderedDict(sorted(bd.items(), key=lambda x: x[1], reverse=False))
            partial_shape_data = x_cv[list(sorted_bd.keys())[:int(len(x_cv) / 3)]]
            # 刚开始shape data为空
            if self.shape_data is None:
                self.shape_data = partial_shape_data
            else:
                self.shape_data = np.vstack((self.shape_data, partial_shape_data))
        # 剔除重复的样本点
        self.shape_data = np.unique(self.shape_data, axis=0)
        self.shape = self._diagram(self.shape_data)
        return self.shape, self.shape_data

    def fast_fit(self, x_data=None, y_data=None):
        for i in range(self.cross_separation):
            if 1 / (self.cross_separation - i) < 1.0:

                x_train, x_data, y_train, y_data = train_test_split(x_data, y_data,
                                                                    train_size=1 / (self.cross_separation - i),
                                                                    random_state=self.random_state)
            else:
                x_train = x_data
                y_train = y_data
            x_train_truth, x_test_truth, y_train_truth, y_test_truth = train_test_split(x_train, y_train,
                                                                                        train_size=0.5,
                                                                                        random_state=self.random_state)
            bd = {}
            for j in range(len(x_test_truth)):
                train_one_test = np.vstack((x_train_truth, x_test_truth[j]))
                bd[j] = self._bottleneck(self._diagram(x_train_truth), self._diagram(train_one_test))
            sorted_bd = collections.OrderedDict(sorted(bd.items(), key=lambda t: t[1], reverse=False))
            partial_shape_data = x_test_truth[list(sorted_bd.keys())[:int(len(x_test_truth) / 4)]]

            if self.shape_data is None:
                self.shape_data = partial_shape_data
            else:
                self.shape_data = np.vstack((self.shape_data, partial_shape_data))

        self.shape_data = np.unique(self.shape_data, axis=0)
        self.shape = self._diagram(self.shape_data)
        return self.shape, self.shape_data

    def fit(self, x_data=None, y_data=None):
        """
        fit data for shape (persistent diagram) and threshold
        :param x_data: point cloud data set
        :param y_data: label, default "o" represent normal
        :return: the shape data or the shape persistent diagram and the threshold
        """
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=self.ratio,
                                                            random_state=self.random_state, shuffle=self.shuffle)
        thresholds = []
        self.shape_data = x_train
        self.shape = self._diagram(self.shape_data)
        # optimization : learning for self.shape
        try:
            with tqdm(range(len(x_test))) as t:
                for i in t:
                    point = x_test[i]
                    data_novelty = np.vstack((self.shape_data, point))
                    diagram_novelty = self._diagram(data_novelty)
                    thresholds.append(np.min([self._bottleneck(self.shape, diagram_novelty), 9.9]))
                mu = np.mean(thresholds)
                sigma = np.std(thresholds, ddof=1)
                self.threshold = mu + self.standard_deviation_coefficient * sigma  # T
                # d-dof = 1 represent unbiased sample standard deviation
        except KeyboardInterrupt:
            t.close()
            raise
        t.close()

        return self.threshold, self.shape, self.shape_data

    def reduce_fit(self, x_data=None, y_data=None):
        """
        将数据分成几个子组，每个子组含有样本点大约30左右，然后约简每个子组的样本点，方法如下：
        对于子组A，依次从A中剔除一个样本点a，得到子组B，计算子组A和子组B之间的拓扑形状距离，如果该距离小于某一个预先定义的阈值，
        则真正的剔除该点a，依次，遍历完A中每一个点，直到A中的点不小于5个为止。
        :param x_data: 训练集
        :param y_data: 训练集的标签，默认都是‘n’，因为只有正常点作为训练集
        :return: 训练集的拓扑骨架集合
        """
        data_number = len(x_data)
        subset_number = data_number // 20
        for i in range(subset_number):
            if 1 / (subset_number - i) < 1.0:
                x_train, x_data, y_train, y_data = train_test_split(x_data, y_data,
                                                                    train_size=1 / (subset_number - i),
                                                                    random_state=self.random_state)
            else:
                x_train = x_data
                y_train = y_data
            number_x_train = len(x_train)
            index = 0
            for j in range(number_x_train):
                distance_reduce = self._bottleneck(self._diagram(x_train), self._diagram(np.delete(x_train, index, 0)))
                # print("distance:", distance_reduce)
                if len(x_train) < 5:
                    # print("len of x_train less than 5")
                    break
                if distance_reduce < self.threshold:
                    x_train = np.delete(x_train, index, 0)
                    index -= 1
                index += 1
            if self.shape_data is None:
                self.shape_data = x_train
            else:
                self.shape_data = np.vstack((self.shape_data, x_train))
        # self.shape_data = np.unique(self.shape_data, axis=0)
        self.shape = self._diagram(self.shape_data)
        return self.shape, self.shape_data

    def grow_fit(self, x_data=None, y_data=None):
        """
        增长模式获得拓扑骨架数据集。
        随机从数据集中选取 5个点作为基，把其他点作为待验证点，如果其他点的加入改变了基数据集的拓扑形状，则认为该点是拓扑骨架数据集的一部分
        更新基，依次进行下去，直到验证完所有的点。
        这里比较重要的是两个超参数，一个是初始基的势，一个是阈值
        :param x_data: 数据集
        :param y_data: 数据集的标签，默认为'n'
        :return: 基
        """
        np.random.seed(self.random_state)
        np.random.shuffle(x_data)
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=6 / len(x_data),
                                                            random_state=self.random_state)
        for i in range(len(x_test)):
            distance = self._bottleneck(self._diagram(x_train), self._diagram(np.vstack((x_train, x_test[i]))))
            if distance > self.threshold:
                x_train = np.vstack((x_train, x_test[i]))
        self.shape_data = x_train
        self.shape = self._diagram(x_train)
        return self.shape_data, self.shape

    def cluster_fit(self, x_data=None, y_data=None, cluster='kmeans', n_cluster=20, branching_factor=100,
                    threshold=1.0, eps=3, min_samples=3, linkage='ward'):
        np.random.seed(self.random_state)
        np.random.shuffle(x_data)
        centroids = np.array([])

        if cluster == 'kmeans':
            clustering = KMeans(n_clusters=n_cluster, random_state=self.random_state).fit(x_data)
            centroids = clustering.cluster_centers_

        elif cluster in ['birch', 'dbscan', 'optics', 'hierarchical', 'spectral', 'tomato']:
            model = Birch(n_clusters=n_cluster, branching_factor=branching_factor, threshold=threshold)
            if cluster == 'dbscan':
                model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=os.cpu_count() - 1)
            elif cluster == 'tomato':
                model = Tomato(density_type="DTM", n_clusters=n_cluster, n_jobs=os.cpu_count() - 1)
            elif cluster == 'optics':
                model = OPTICS(eps=eps, min_samples=min_samples, n_jobs=os.cpu_count() - 1)
            elif cluster == 'hierarchical':
                model = AgglomerativeClustering(n_clusters=n_cluster, linkage=linkage)
            elif cluster == 'spectral':
                model = SpectralClustering(n_clusters=n_cluster, assign_labels="discretize", eigen_solver='arpack',
                                           affinity="nearest_neighbors", random_state=self.random_state,
                                           n_jobs=os.cpu_count() - 1)
            clustering = model.fit(x_data)
            labels = clustering.labels_
            unique_labels = np.unique(labels)

            if len(unique_labels) < 3:
                return 0
            for i, label in zip(range(len(unique_labels)), unique_labels):
                if i == 0:
                    centroids1 = np.median(x_data[labels == label], axis=0)
                    centroids2 = np.mean(x_data[labels == label], axis=0)
                    centroids = np.vstack([centroids1, centroids2])
                else:
                    centroids1 = np.median(x_data[labels == label], axis=0)
                    centroids2 = np.mean(x_data[labels == label], axis=0)
                    new_center = np.vstack([centroids1, centroids2])
                    centroids = np.vstack([centroids, new_center])

        self.shape_data = centroids
        self.shape = self._diagram(centroids)
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
        self.predict(x_test=x_test)
        return self.scores

    # def OX_fit(self, X, output_path='.'):
    #     '''
    #     param x_train: the train set
    #     param output_path: save the base shape dataset B to output path
    #     '''
    #     self.data = X
    #     data = pd.DataFrame(data=X)
    #     base_list = list(data.index)
    #     abandom_point = []
    #
    #     if self.base > len(data):
    #         print("the base shape data set is too small")
    #     else:
    #         try:
    #             with tqdm(np.arange(len(data.index))) as t:
    #                 for i in t:
    #                     choice_index = i
    #                     choice_point = np.array(data.iloc[choice_index, :])
    #                     temp_list = copy.deepcopy(base_list)
    #                     base_list.remove(choice_index)
    #                     self.data = np.array(data.iloc[base_list, :])
    #                     if len(self.data) < self.base:
    #                         break
    #                     else:
    #                         if self._bottleneck(choice_point) < self.threshold:
    #                             base_list = copy.deepcopy(temp_list)
    #                         else:
    #                             abandom_point.append(choice_index)
    #                             continue
    #         except KeyboardInterrupt:
    #             t.close()
    #             raise
    #
    #         t.close()
    #
    #         B = data.iloc[base_list, :]
    #         self.data = np.array(B)  # the model is being fitted by train set
    #         B.to_csv(output_path + ".csv")
    #         print("self data shape: ", data.shape[0])
    #         print("B data shape: ", B.shape)
    #         with open(output_path + ".txt", 'w') as f:  # write abandom point to file
    #             for item in abandom_point:
    #                 f.write("%s\n" % item)
