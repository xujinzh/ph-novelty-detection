#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Jinzhong Xu
# @Contact : jinzhongxu@csu.ac.cn
# @Time    : 2020/11/27 17:06
# @File    : comparison.py
# @Software: PyCharm


from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from tda import roc


def classical(x_train, x_test, y_test):
    # lof novelty detection
    clf = LocalOutlierFactor(novelty=True)
    clf.fit(x_train)
    y_scores = clf.score_samples(x_test)
    auc_lof = roc.area(y_test=y_test, y_scores=y_scores, pos_label=1, title='LOF - ', plot_roc=False)

    # svm novelty detection
    clf = OneClassSVM(kernel="rbf", gamma='auto')
    clf.fit(x_train)
    y_scores = clf.score_samples(x_test)
    auc_svm = roc.area(y_test=y_test, y_scores=y_scores, pos_label=1, title='OC-SVM - ', plot_roc=False)

    return auc_lof, auc_svm
