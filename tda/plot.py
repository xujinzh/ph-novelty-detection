#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Jinzhong Xu
# @Contact : jinzhongxu@csu.ac.cn
# @Time    : 2020/11/27 17:35
# @File    : plot.py
# @Software: PyCharm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from pathlib import Path


def draw(auc_classical, auc_ph, file_name):
    lof_score, svm_score = auc_classical

    ph_scores = [x[1] for x in auc_ph]
    ph_argument = [x[0] for x in auc_ph]
    len_ph_score = len(ph_scores)
    lof_scores = [lof_score] * len_ph_score
    svm_scores = [svm_score] * len_ph_score

    data_name = file_name.split('-')[0]
    cluster = auc_ph[0][0][0]

    x = np.linspace(0, 1, len_ph_score)
    (lof_plot,) = plt.plot(x, lof_scores, "g+")
    (svm_plot,) = plt.plot(x, svm_scores, "ro")
    (ph_plot,) = plt.plot(x, ph_scores, "b^")
    plt.legend([svm_plot, lof_plot, ph_plot], ["svm", "lof", "ph"])

    # 画图显示
    if cluster == 'birch':
        cluster, n_cluster, branching_factor, cluster_threshold = auc_ph[0][0]
        plt.title(
            "cluster={0}, n_clusters={1}, branching_factor={2}, cluster_threshold={3}".format(cluster, n_cluster,
                                                                                              branching_factor,
                                                                                              cluster_threshold))
        plt.savefig(
            "./output/data={0}-cluster={1}-n_clusters={2}-branching_factor={3}-cluster_threshold={4}.png".format(
                data_name, cluster, n_cluster, branching_factor, cluster_threshold))
    elif cluster in ['kmeans', 'tomato', 'spectral']:
        cluster, n_cluster = auc_ph[0][0]
        plt.title("cluster={0}, n_clusters={1}".format(cluster, n_cluster))
        plt.savefig(
            "./output/data={0}-cluster={1}-n_clusters={2}.png".format(data_name, cluster, n_cluster))
    elif cluster == 'hierarchical':
        cluster, n_cluster, linkage = auc_ph[0][0]
        plt.title("cluster={0}, n_clusters={1}, linkage={2}".format(cluster, n_cluster, linkage))
        plt.savefig(
            "./output/data={0}-cluster={1}-n_clusters={2}-linkage={3}.png".format(data_name, cluster,
                                                                                  n_cluster, linkage))
    elif cluster in ['dbscan', 'optics']:
        cluster, eps, min_samples = auc_ph[0][0]
        plt.title("cluster={0}, eps={1}, min_samples={2}".format(cluster, eps, min_samples))
        plt.savefig(
            "./output/data={0}-cluster={1}-eps={2}-min_samples={3}.png".format(data_name, cluster,
                                                                               eps, min_samples))
    plt.close()

    # 把结果保存到文件
    score_dict = {'argument': ph_argument, 'ph': ph_scores, 'lof': lof_scores, 'svm': svm_scores}
    csv_file = './output/data={0}-cluster={1}.csv'.format(data_name, cluster)

    df = pd.DataFrame(score_dict)

    if not os.path.exists(csv_file):
        Path(csv_file).touch()
    df.to_csv(csv_file)
