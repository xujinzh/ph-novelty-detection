#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Jinzhong Xu
# @Contact : jinzhongxu@csu.ac.cn
# @Time    : 2020/8/21 18:16
# @File    : model.py
# @Software: PyCharm

from tda import topology as top
from tda import roc


def PersistentHomology(x_train, y_train, x_test, y_test, max_dimension=3, ratio=0.3, standard_deviation_coefficient=3,
                       shuffle=False, max_edge_length=6, cross_separation=3, random_state=3, threshold=0.45,
                       cluster='spectral', n_cluster=25, eps=15, min_samples=4, branching_factor=20,
                       cluster_threshold=0.8, linkage='ward', e=0.0, sparse=1.0, plot_roc=False):
    clf = top.PHNovDet(max_dimension=max_dimension, ratio=ratio, e=e, sparse=sparse,
                       standard_deviation_coefficient=standard_deviation_coefficient, shuffle=shuffle,
                       max_edge_length=max_edge_length, cross_separation=cross_separation,
                       random_state=random_state, threshold=threshold)

    result = clf.fit(x_train, y_train, cluster=cluster, n_cluster=n_cluster, branching_factor=branching_factor,
                     threshold=cluster_threshold, eps=eps, min_samples=min_samples, linkage=linkage)

    if result == 0:
        return 0.5

    # 预测分数
    y_scores = clf.score_samples(x_test)

    # 化成roc曲线，返回auc值
    auc_ph = roc.area(y_test=y_test, y_scores=y_scores, pos_label=-1, title='PH - ', plot_roc=plot_roc)

    return auc_ph
