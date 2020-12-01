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
from tda.comparison import classical
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import roc_auc_score
import random
from tda.model import PersistentHomology
import matplotlib.pyplot as plt
import os
from tqdm import trange, tqdm
from tda.preprocessing import prepare_data
from tda.plot import draw


def just_do_it(path, cluster, multiple, random_state):
    # 获取数据集的名字
    file_name = os.path.split(path)[1]

    normals, normal_labels, x_test, y_test = prepare_data(path, multiple=multiple, random_state=random_state)

    # 比较算法 lof and svm
    auc_classical = classical(x_train=normals, x_test=x_test, y_test=y_test)

    auc_ph = []
    print("\t正在使用聚类算法 {0} 处理数据集 {1} ...".format(cluster, file_name.split('-')[0]))

    if cluster in ['spectral', 'kmeans', 'tomato']:
        for n_cluster in trange(10, 30):
            ph = PersistentHomology(x_train=normals, x_test=x_test, y_train=normal_labels, y_test=y_test,
                                    cluster=cluster, n_cluster=n_cluster, random_state=random_state)
            auc_ph.append([(cluster, n_cluster), ph])
    elif cluster == 'hierarchical':
        for n_cluster in trange(10, 30):
            for linkage in ['ward', 'complete', 'average', 'single']:
                ph = PersistentHomology(x_train=normals, x_test=x_test, y_train=normal_labels, y_test=y_test,
                                        cluster=cluster, n_cluster=n_cluster, linkage=linkage,
                                        random_state=random_state)
                auc_ph.append([(cluster, n_cluster, linkage), ph])
    elif cluster in ['dbscan', 'optics']:
        for eps in trange(5, 15):
            for min_samples in range(3, 9):
                ph = PersistentHomology(x_train=normals, x_test=x_test, y_train=normal_labels, y_test=y_test,
                                        cluster=cluster, eps=eps, min_samples=min_samples, random_state=random_state)
                auc_ph.append([(cluster, eps, min_samples), ph])
    elif cluster == 'birch':
        for n_cluster in trange(10, 30):
            for branching_factor in range(5, 25):
                for cluster_threshold in np.arange(0.5, 0.9, 0.2):
                    ph = PersistentHomology(x_train=normals, x_test=x_test, y_train=normal_labels, y_test=y_test,
                                            cluster=cluster, n_cluster=n_cluster, branching_factor=branching_factor,
                                            cluster_threshold=cluster_threshold, random_state=random_state)
                    auc_ph.append([(cluster, n_cluster, branching_factor, cluster_threshold), ph])

    print("\t聚类算法 {0} 完成数据集 {1} 的处理工作 ^_^".format(cluster, file_name.split('-')[0]))

    draw(auc_classical, auc_ph, file_name, os.path.split(path)[0])
