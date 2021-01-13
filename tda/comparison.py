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
from persim import bottleneck, sliced_wasserstein
from ripser import ripser
import numpy as np
from tda.preprocessing import prepare_data
from tqdm import trange
import warnings

warnings.simplefilter("ignore")


def classical(x_train, x_test, y_test, plot_roc=False):
    # lof novelty detection
    clf = LocalOutlierFactor(novelty=True)
    clf.fit(x_train)
    y_scores = clf.score_samples(x_test)
    auc_lof = roc.area(y_test=y_test, y_scores=y_scores, pos_label=1, title='LOF - ', plot_roc=plot_roc)

    # svm novelty detection
    clf = OneClassSVM(kernel="rbf", gamma='auto')
    clf.fit(x_train)
    y_scores = clf.score_samples(x_test)
    auc_svm = roc.area(y_test=y_test, y_scores=y_scores, pos_label=1, title='OC-SVM - ', plot_roc=plot_roc)

    return auc_lof, auc_svm


if __name__ == '__main__':
    path = '../data/penGlobal-unsupervised-ad.csv'
    normals, normal_labels, x_test, y_test = prepare_data(path=path, multiple=1, random_state=3)
    # scikit-tda
    auc_tda = {}
    for n in trange(75, 550):
        res = ripser(normals, n_perm=n)
        dgms_res = res["dgms"][1]
        idx_res = res["idx_perm"]
        data_res = normals[idx_res]
        y_scores = []
        for i in range(len(x_test)):
            data_nov = np.vstack((data_res, x_test[i]))
            dgms_nov = ripser(data_nov)["dgms"][1]
            y_scores.append(sliced_wasserstein(dgms_nov, dgms_res, M=5))
        auc = roc.area(y_test=y_test, y_scores=y_scores, pos_label=1, title='Scikit-TDA - ', plot_roc=False)
        if auc > 0.95:
            auc_tda[n] = auc
    print(auc_tda)
