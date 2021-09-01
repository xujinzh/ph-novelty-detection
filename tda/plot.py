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
from tda import comparison
from tda import preprocessing



def draw(auc_classical, auc_ph, file_name, path):
    lof_score, svm_score = auc_classical
    auc_ph.sort(reverse=True, key=lambda x: x[1])
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

    plt.title("cluster={0}, data={1}".format(cluster, data_name))
    plt.savefig(path + "/../output/data={0}-cluster={1}.png".format(data_name, cluster))
    plt.close()

    # 把结果保存到文件
    score_dict = {'argument': ph_argument, 'ph': ph_scores, 'lof': lof_scores, 'svm': svm_scores}
    csv_file = path + '/../output/data={0}-cluster={1}.csv'.format(data_name, cluster)

    df = pd.DataFrame(score_dict)

    if not os.path.exists(csv_file):
        Path(csv_file).touch()
    df.to_csv(csv_file)


def drawMul(auc_classical, auc_phs, file_name, path):
    auc_lof, auc_svm = auc_classical
    auc_phs.sort(reverse=True, key=lambda x: x[1])
    auc_lof.sort(reverse=True, key=lambda x: x[1])
    auc_svm.sort(reverse=True, key=lambda x: x[1])

    phs_len = len(auc_phs)
    lof_len = len(auc_lof)
    svm_len = len(auc_svm)
    max_len = max(phs_len, lof_len, svm_len)

    auc_phs_value = [x[1] for x in auc_phs]
    ph_argument = [x[0] for x in auc_phs]
    auc_lof_value = [x[1] for x in auc_lof]
    lof_argument = [x[0] for x in auc_lof]
    auc_svm_value = [x[1] for x in auc_svm]
    svm_argument = [x[0] for x in auc_svm]

    auc_phs_value.extend((max_len - phs_len) * [auc_phs_value[-1]])
    auc_lof_value.extend((max_len - lof_len) * [auc_lof_value[-1]])
    auc_svm_value.extend((max_len - svm_len) * [auc_svm_value[-1]])

    ph_argument.extend((max_len - phs_len) * [ph_argument[-1]])
    lof_argument.extend((max_len - lof_len) * [lof_argument[-1]])
    svm_argument.extend((max_len - svm_len) * [svm_argument[-1]])

    data_name = file_name.split('-')[0]
    cluster = auc_phs[0][0][0]

    x = np.linspace(0, 1, max_len)
    (lof_plot,) = plt.plot(x, auc_lof_value, "g+")
    (svm_plot,) = plt.plot(x, auc_svm_value, "ro")
    (ph_plot,) = plt.plot(x, auc_phs_value, "b^")
    plt.legend([svm_plot, lof_plot, ph_plot], ["svm", "lof", "ph"])

    plt.title("cluster={0}, data={1}".format(cluster, data_name))
    plt.savefig(path + "/../output/data={0}-cluster={1}.png".format(data_name, cluster))
    plt.close()

    # 把结果保存到文件
    score_dict = {'arg_ph': ph_argument, 'ph': auc_phs_value, 'arg_lof': lof_argument, 'lof': auc_lof_value,
                  'arg_svm': svm_argument, 'svm': auc_svm_value}
    csv_file = path + '/../output/data={0}-cluster={1}.csv'.format(data_name, cluster)

    df = pd.DataFrame(score_dict)

    if not os.path.exists(csv_file):
        Path(csv_file).touch()
    df.to_csv(csv_file)


if __name__ == '__main__':
    path = '../data/breast-cancer-unsupervised-ad.csv'
    normals, normal_labels, x_test, y_test = preprocessing.prepare_data(path=path, multiple=1, random_state=3)
    filename = os.path.basename(path)
    dirs = os.path.dirname(path)
    cls = comparison.classical(x_train=normals, x_test=x_test, y_test=y_test)
    phs = [[('tomato', 1), 0.9], [('kmeans', 3), 0.89], [('birch', 0.5, 32), 0.99]]
    drawMul(auc_classical=cls, auc_phs=phs, file_name=filename, path=dirs)
