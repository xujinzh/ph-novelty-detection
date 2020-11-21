#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Jinzhong Xu
# @Contact : jinzhongxu@csu.ac.cn
# @Time    : 2020/8/21 18:16
# @File    : model.py
# @Software: PyCharm

from tda import topology as top
from tda import roc
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


def multi_model(x_train, y_train, x_test, y_test, max_dimension=2, ratio=0.3, standard_deviation_coefficient=3,
                shuffle=False, max_edge_length=6, cross_separation=3, random_state=3, threshold=0.45):
    # PH novelty detection

    clf = top.PHNovDet(max_dimension=max_dimension, ratio=ratio,
                       standard_deviation_coefficient=standard_deviation_coefficient, shuffle=shuffle,
                       max_edge_length=max_edge_length, cross_separation=cross_separation,
                       random_state=random_state, threshold=threshold)
    # clf.grow_fit(x_train, y_train)
    # clusters = [['birch', 'n_cluster', 'branching_factor', 'threshold'], ['kmeans', 'n_cluster'],
    #             ['dbscan', 'eps', 'min_samples']]

    cluster = 'dbscan'
    n_cluster = 20
    branching_factor = 20
    threshold = 0.8
    eps = 15
    min_samples = 4
    clf.cluster_fit(x_train, y_train, cluster=cluster, n_cluster=n_cluster, branching_factor=branching_factor,
                    threshold=threshold, eps=eps, min_samples=min_samples)

    # 预测结果标签
    ph_predicted = clf.predict(x_test)

    # 预测分数
    y_scores = clf.score_samples(x_test)

    # 化成roc曲线，返回auc值
    auc_ph = roc.area(y_test=y_test, y_scores=y_scores, pos_label=-1, title='PH - ', plot_roc=False)
    # auc_ph = roc_auc_score(y_true=y_test, y_score=y_scores)

    clf = LocalOutlierFactor(novelty=True, n_neighbors=9)

    clf.fit(x_train)

    rof_predicted = clf.predict(x_test)

    y_scores = clf.score_samples(x_test)

    auc_lof = roc.area(y_test=y_test, y_scores=y_scores, pos_label=1, title='LOF - ', plot_roc=False)

    clf = OneClassSVM(kernel="rbf", gamma='auto')
    clf.fit(x_train)

    svm_predicted = clf.predict(x_test)

    y_scores = clf.score_samples(x_test)

    auc_svm = roc.area(y_test=y_test, y_scores=y_scores, pos_label=1, title='OC-SVM - ', plot_roc=False)

    # if auc_ph == 1.0:
    #     print('*' * 60)
    #     print("auc of ph: %.3f" % auc_ph)
    #     print("auc of lof:%.3f" % auc_lof)
    #     print("auc of svm:%.3f" % auc_svm)
    #     print('*' * 60)

    print('=' * 60)
    print("auc of ph: %.3f" % auc_ph)
    print("auc of lof:%.3f" % auc_lof)
    print("auc of svm:%.3f" % auc_svm)
    return auc_svm, auc_lof, auc_ph, cluster, n_cluster, branching_factor, threshold, eps, min_samples
