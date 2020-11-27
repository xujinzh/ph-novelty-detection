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
import os
import random


def just_do_it(path, cluster='dbscan', n_cluster=25, eps=15, min_samples=5, branching_factor=20, cluster_threshold=0.8,
               linkage='ward', random_state=3):
    # 获取数据集的名字
    file_name = os.path.split(path)[1]
    auc = []
    multiple = 1
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

    random.seed(random_state)
    random.shuffle(normals)

    # 将测试集中添加异常点，成为真正的测试集
    x_test = np.vstack((outliers, normals[: multiple * len(outliers)]))
    y_test = np.hstack((outlier_labels, normal_labels[: multiple * len(outlier_labels)]))
    # y_test[y_test == 'o'] = -1; y_test[y_test == 'n'] = 1
    # 把y_test中满足条件 y_test == 'o'的输出为-1，其他输出为1。最后，转化为list，方便计算roc
    y_test = list(np.where(y_test == 'o', -1, 1))

    # 剩余的正常点用于训练模型
    normals = normals[multiple * len(outliers):]
    normal_labels = normal_labels[multiple * len(outlier_labels):]

    for train_size in np.arange(0.65, 0.95, 0.02):
        train_size = round(train_size, 2)
        # 划分其他为训练集和验证集
        x_train, x_cv, y_train, y_cv = train_test_split(normals, normal_labels, train_size=train_size,
                                                        random_state=random_state)

        auc_score = multi_model(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, threshold=0.5,
                                cluster=cluster, n_cluster=n_cluster, eps=eps, min_samples=min_samples,
                                branching_factor=branching_factor, cluster_threshold=cluster_threshold,
                                linkage=linkage, random_state=random_state)
        auc.append(auc_score)

    svm_score = []
    lof_score = []
    ph_score = []
    for score in auc:
        svm_score.append(score[0])
        lof_score.append(score[1])
        ph_score.append(score[2])

    mean_svm = np.mean(svm_score)
    mean_lof = np.mean(lof_score)
    mean_ph = np.mean(ph_score)

    if mean_ph > mean_lof or mean_ph > mean_svm:

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
        elif cluster in ['kmeans', 'hierarchical', 'spectral']:
            plt.title("cluster={0}, n_clusters={1}".format(cluster, n_cluster))
        elif cluster == 'dbscan':
            plt.title("cluster={0}, eps={1}, min_samples={2}".format(cluster, eps, min_samples))
        elif cluster == 'optics':
            plt.title("cluster={0}, eps={1}, min_samples={2}".format(cluster, eps, min_samples))
        plt.savefig(
            "./output/data={0}-cluster={1}-n_clusters={2}-branching_factor={3}-threshold={4}.png".format(
                file_name.split('-')[0], cluster, n_cluster,
                branching_factor,
                threshold))
        plt.close()
