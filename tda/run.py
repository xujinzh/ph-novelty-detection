#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Jinzhong Xu
# @Contact : jinzhongxu@csu.ac.cn
# @Time    : 2020/8/21 18:23
# @File    : run.py
# @Software: PyCharm

import numpy as np
import pandas as pd
from tda import topology as top
from tda import roc
import copy
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import roc_auc_score
import random
from tda.model import multi_model
import matplotlib.pyplot as plt


def just_do_it(path):
    # 读取数据
    data = pd.read_csv(path, header=None)

    # 获取异常数据和正常数据
    outlier_data = data[data.iloc[:, -1] == 'o']
    normal_data = data[data.iloc[:, -1] == 'n']

    # 提取异常点到ndarray和异常点标签
    outliers = np.array(outlier_data.iloc[:, :-1])
    outlier_labels = np.array(outlier_data.iloc[:, -1])

    # 提取正常点到ndarray和正常点标签
    normals = np.array(normal_data.iloc[:, :-1])
    normal_labels = np.array(normal_data.iloc[:, -1])

    auc = []

    for outer_train_size in np.arange(0.8, 0.99, 0.01):
        outer_train_size = round(outer_train_size, 2)
        # 划分正常样本为测试集和其他（用于再次划分为训练集和验证集）
        svm_score, x_test, lof_score, y_test = train_test_split(normals, normal_labels, train_size=outer_train_size,
                                                                random_state=3)

        # 将测试集中添加异常点，称为真正的测试集
        x_test = np.vstack((outliers, x_test))
        y_test = np.hstack((outlier_labels, y_test))
        # y_test[y_test == 'o'] = -1; y_test[y_test == 'n'] = 1
        # 把y_test中满足条件 y_test == 'o'的输出为-1，其他输出为1。最后，转化为list，方便计算roc
        y_test = list(np.where(y_test == 'o', -1, 1))

        for train_size in np.arange(0.8, 0.99, 0.01):
            train_size = round(train_size, 2)
            # 划分其他为训练集和验证集
            x_train, x_cv, y_train, y_cv = train_test_split(svm_score, lof_score, train_size=train_size, random_state=3)

            auc_score = multi_model(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, threshold=0.5)
            auc.append(auc_score)

    svm_score = []
    lof_score = []
    ph_score = []
    for score in auc:
        svm_score.append(score[0])
        lof_score.append(score[1])
        ph_score.append(score[2])

    x = np.linspace(0, 1, len(ph_score))
    svm_plot, = plt.plot(x, svm_score, 'ro')
    lof_plot, = plt.plot(x, lof_score, 'g+')
    ph_plot, = plt.plot(x, ph_score, 'b^')
    plt.legend([svm_plot, lof_plot, ph_plot], ['svm', 'lof', 'ph'])

    cluster = auc[0][3]
    n_cluster = auc[0][4]
    branching_factor = auc[0][5]
    threshold = auc[0][6]
    eps = auc[0][7]
    min_samples = auc[0][8]

    if cluster == 'birch':
        plt.title(
            "cluster={0}, n_clusters={1}, branching_factor={2}, threshold={3}".format(cluster, n_cluster,
                                                                                      branching_factor,
                                                                                      threshold))
    elif cluster == 'kmeans':
        plt.title("cluster={0}, n_clusters={1}".format(cluster, n_cluster))
    elif cluster == 'dbscan':
        plt.title("cluster={0}, eps={1}, min_samples={2}".format(cluster, eps, min_samples))
    plt.show()
