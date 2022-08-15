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
    ap.add_argument('-d', '--data', required=False, type=str,
                    default='data/breast-cancer-unsupervised-ad.csv',
                    help='data path')
    ap.add_argument('-m', '--multiple', required=False, type=float, default=1.0,
                    help='normal point than novelty point in test data')
    ap.add_argument('-c', '--cluster', required=False, type=str,
                    default='tomato', help='clutering method, include '
                                           'tomato(n_cluster), '
                                           'kmeans(n_cluster), '
                                           'hierarchical(n_cluster, linkage), '
                                           'spectral(n_cluster), '
                                           'birch(n_cluster, branching_factor, cluster_threshold), '
                                           'dbscan(n_cluster, eps, min_samples), '
                                           'optics(n_cluster, eps, min_samples)')
    ap.add_argument('-n', '--n_cluster', required=False, type=int, default=20,
                    help='number of centroid')
    ap.add_argument('-r', '--random', required=False, type=int, default=3, help='random state')
    ap.add_argument('-l', '--linkage', required=False, type=str, default='ward',
                    help='linkage for hierarchical')
    ap.add_argument('-b', '--branching_factor', required=False, type=int, default=3,
                    help='branching_factor for birch')
    ap.add_argument('-t', '--cluster_threshold', required=False, type=float, default=0.3,
                    help='cluster_threshold for birch')
    ap.add_argument('--eps', required=False, type=int, default=2,
                    help='eps for dbscan or optics cluster')
    ap.add_argument('--min_samples', required=False, type=int, default=2,
                    help='min_samples for dbscan or optics cluster')
    ap.add_argument('-p', '--plot_roc', required=False, type=bool, default=True, help='plot roc')
    ap.add_argument('-s', '--sparse', required=False, type=float, default=None,
                    help='sparse for construct complex')
    ap.add_argument('-g', '--use_gpu', required=False, type=str, default='yes',
                    help='use GPU for "yes" or CPU for "no"')

    args = vars(ap.parse_args())

    # 读取数据
    data_path = args['data']
    multiple = args['multiple']
    cluster = args['cluster']
    n_cluster = args['n_cluster']
    random_state = args['random']
    linkage = args['linkage']
    branching_factor = args['branching_factor']
    cluster_threshold = args['cluster_threshold']
    eps = args['eps']
    min_samples = args['min_samples']
    plot_roc = args['plot_roc']
    sparse = args['sparse']
    use_gpu = args['use_gpu']

    print("开始准备数据")
    normals, normal_labels, x_test, y_test = prepare_data(data_path, multiple=multiple,
                                                          random_state=random_state)


    # # 比较算法 lof 和 oneclass-svm
    # classical(x_train=normals, x_test=x_test, y_test=y_test, plot_roc=plot_roc)

    print("开始使用拓扑持续同调方法进行新颖点检测")
    PersistentHomology(x_train=normals, x_test=x_test, y_train=normal_labels, y_test=y_test,
                       random_state=random_state, cluster=cluster, n_cluster=n_cluster,
                       linkage=linkage, branching_factor=branching_factor,
                       cluster_threshold=cluster_threshold, eps=eps, min_samples=min_samples,
                       sparse=sparse, plot_roc=plot_roc, use_gpu=use_gpu)
