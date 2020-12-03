#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Jinzhong Xu
# @Contact : jinzhongxu@csu.ac.cn
# @Time    : 2020/11/27 17:10
# @File    : preprocessing.py
# @Software: PyCharm

import numpy as np
import pandas as pd
import random


def prepare_data(path, multiple=2, random_state=3):
    # 读取数据
    data = pd.read_csv(path, header=None)

    # 获取异常数据和正常数据
    outlier_data = data[data.iloc[:, -1] == 'o']
    normal_data = data[data.iloc[:, -1] == 'n']

    # 提取异常点到 ndarray 和异常点标签
    outliers = np.array(outlier_data.iloc[:, :-1])
    outlier_labels = np.array(outlier_data.iloc[:, -1])

    # 提取正常点到 ndarray 和正常点标签
    normals = np.array(normal_data.iloc[:, :-1])
    normal_labels = np.array(normal_data.iloc[:, -1])

    random.seed(random_state)
    random.shuffle(normals)

    # 截断正常样本点，和异常点组成测试集
    cutoff = int(multiple * len(outliers))

    # 将测试集中添加异常点，成为真正的测试集
    x_test = np.vstack((outliers, normals[: cutoff]))
    y_test = np.hstack((outlier_labels, normal_labels[: cutoff]))
    # y_test[y_test == 'o'] = -1; y_test[y_test == 'n'] = 1
    # 把y_test中满足条件 y_test == 'o'的输出为-1，其他输出为1。最后，转化为list，方便计算roc
    y_test = list(np.where(y_test == 'o', -1, 1))

    # 剩余的正常点用于训练模型
    normals = normals[cutoff:]
    normal_labels = normal_labels[cutoff:]

    return normals, normal_labels, x_test, y_test
