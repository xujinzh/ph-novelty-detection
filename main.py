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
from tda.timestamps import display_time
from multiprocessing import Pool
import warnings

warnings.filterwarnings("ignore")


def main(cluster):
    # 读取数据
    data_paths = ['./data/breast-cancer-unsupervised-ad.csv', './data/letter-unsupervised-ad.csv',
                  './data/pen-global-unsupervised-ad.csv', './data/annthyroid-unsupervised-ad.csv',
                  './data/satellite-unsupervised-ad.csv']

    for data_path in data_paths:
        file_name = os.path.split(data_path)[1]
        print("\t正在使用聚类算法 {0} 处理数据集 {1} ...".format(cluster, file_name.split('-')[0]))
        if cluster == 'spectral' or cluster == 'kmeans':
            for n_cluster in range(10, 38):
                run.just_do_it(path=data_path, cluster=cluster, n_cluster=n_cluster)
        elif cluster == 'hierarchical':
            for n_cluster in range(10, 38):
                for linkage in ['ward', 'complete', 'average', 'single']:
                    run.just_do_it(path=data_path, cluster=cluster, linkage=linkage, n_cluster=n_cluster)
        elif cluster == 'dbscan' or cluster == 'optics':
            for eps in range(5, 20):
                for min_samples in range(3, 9):
                    run.just_do_it(path=data_path, cluster=cluster, eps=eps, min_samples=min_samples)
        elif cluster == 'birch':
            for n_cluster in range(10, 38):
                for branching_factor in range(5, 50):
                    for cluster_threshold in np.arange(0.5, 0.9, 0.1):
                        run.just_do_it(path=data_path, cluster=cluster, n_cluster=n_cluster,
                                       branching_factor=branching_factor, cluster_threshold=cluster_threshold)
        print("\t聚类算法 {0} 完成数据集 {1} 的处理工作 ^_^".format(cluster, file_name.split('-')[0]))


if __name__ == '__main__':
    clusters = ['spectral', 'hierarchical', 'kmeans', 'dbscan', 'optics', 'birch']
    # for cluster in clusters:
    #     print("使用聚类方法 {0} 进行处理 ...".format(cluster))
    #     main(cluster=cluster)
    #     print("\t聚类方法 {0} 处理结束 ^_^".format(cluster))
    # print("整个任务完成！")
    with Pool(6) as p:
        p.map(main, clusters)
    print("整个任务完成！")
