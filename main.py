#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Jinzhong Xu
# @Contact : jinzhongxu@csu.ac.cn
# @Time    : 2020/8/21 18:27
# @File    : main.py
# @Software: PyCharm

from tda import run
import numpy as np
import os


def main(cluster):
    # 读取数据
    data_paths = ['./data/breast-cancer-unsupervised-ad.csv', './data/letter-unsupervised-ad.csv',
                  './data/pen-global-unsupervised-ad.csv', './data/annthyroid-unsupervised-ad.csv',
                  './data/satellite-unsupervised-ad.csv']

    for data_path in data_paths:
        file_name = os.path.split(data_path)[1]
        print("\t正在处理数据集 {} ...".format(file_name.split('-')[0]))
        if cluster == 'spectral' or cluster == 'hierarchical' or cluster == 'kmeans':
            for n_cluster in range(8, 130):
                run.just_do_it(path=data_path, cluster=cluster, n_cluster=n_cluster)
        elif cluster == 'hierarchical':
            for n_cluster in range(8, 130):
                for linkage in ['ward', 'complete', 'average', 'single']:
                    run.just_do_it(path=data_path, cluster=cluster, linkage=linkage)
        elif cluster == 'dbscan' or cluster == 'optics':
            for eps in range(5, 25):
                for min_samples in range(3, 10):
                    run.just_do_it(path=data_path, cluster=cluster, eps=eps, min_samples=min_samples)
        elif cluster == 'birch':
            #  'n_cluster', 'branching_factor', 'threshold'
            for n_cluster in range(8, 130):
                for branching_factor in range(5, 25):
                    for cluster_threshold in np.arange(0.5, 1.0, 0.1):
                        run.just_do_it(path=data_path, cluster=cluster, n_cluster=n_cluster,
                                       branching_factor=branching_factor, cluster_threshold=cluster_threshold)


if __name__ == '__main__':
    clusters = ['spectral', 'hierarchical', 'kmeans', 'dbscan', 'optics', 'birch']
    for cluster in clusters:
        print("使用聚类方法 {0} 进行处理...".format(cluster))
        main(cluster=cluster)
