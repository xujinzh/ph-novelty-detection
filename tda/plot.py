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

    plt.title("cluster={0}, data={1}".format(cluster, data_name))
    plt.savefig("./output/data={0}-cluster={1}.png".format(data_name, cluster))
    plt.close()

    # 把结果保存到文件
    score_dict = {'argument': ph_argument, 'ph': ph_scores, 'lof': lof_scores, 'svm': svm_scores}
    csv_file = './output/data={0}-cluster={1}.csv'.format(data_name, cluster)

    df = pd.DataFrame(score_dict)

    if not os.path.exists(csv_file):
        Path(csv_file).touch()
    df.to_csv(csv_file)
