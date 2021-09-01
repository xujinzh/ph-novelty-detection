#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Jinzhong Xu
# @Contact : jinzhongxu@csu.ac.cn
# @File    : run.py
# @Time    : 2021/1/4 18:19
# @Software: PyCharm

from tda.comparison import classical
from tda.model import PersistentHomology
from tda.preprocessing import prepare_data
import argparse


def just_do_it():
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--data', required=True, type=str, default='./data/breast-cancer-unsupervised-ad.csv',
                    help='data path')
    ap.add_argument('-m', '--multiple', required=False, type=float, default=1.0,
                    help='normal point than novelty point in test data')
    ap.add_argument('-c', '--cluster', required=False, type=str, default='tomato', help='clutering method')
    ap.add_argument('-n', '--nClusters', required=False, type=int, default=9, help='number of clusters')
    ap.add_argument('-r', '--random', required=False, type=int, default=3, help='random state')
    ap.add_argument('-l', '--linkage', required=False, type=str, default='ward', help='linkage for hierarchical')
    ap.add_argument('-b', '--branchingFactor', required=False, type=int, default=3, help='branching_factor for birch')
    ap.add_argument('-t', '--clusterThreshold', required=False, type=float, default=0.3,
                    help='cluster_threshold for birch')
    ap.add_argument('-p', '--plotRoc', required=False, type=bool, default=True, help='plot roc')
    ap.add_argument('-s', '--sparse', required=False, type=float, default=None, help='sparse for construct complex')

    args = vars(ap.parse_args())

    # 读取数据
    data_path = args['data']
    multiple = args['multiple']
    cluster = args['cluster']
    n_cluster = args['nClusters']
    random_state = args['random']
    linkage = args['linkage']
    branching_factor = args['branchingFactor']
    cluster_threshold = args['clusterThreshold']
    plot_roc = args['plotRoc']
    sparse = args['sparse']

    normals, normal_labels, x_test, y_test = prepare_data(data_path, multiple=multiple, random_state=random_state)

    # 比较算法 lof 和 oneclass-svm
    classical(x_train=normals, x_test=x_test, y_test=y_test, plot_roc=plot_roc)

    PersistentHomology(x_train=normals, x_test=x_test, y_train=normal_labels, y_test=y_test, random_state=random_state,
                       cluster=cluster, n_cluster=n_cluster, linkage=linkage, branching_factor=branching_factor,
                       cluster_threshold=cluster_threshold, sparse=sparse, plot_roc=plot_roc)
