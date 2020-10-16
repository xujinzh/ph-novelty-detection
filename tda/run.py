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


def just_do_it(path):
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

    for outer_train_size in np.arange(0.88, 1, 0.01):
        outer_train_size = round(outer_train_size, 2)
        # 划分正常样本为测试集和其他（用于再次划分为训练集和验证集）
        x, x_test, y, y_test = train_test_split(normals, normal_labels, train_size=outer_train_size, random_state=3)

        # 将测试集中添加异常点，称为真正的测试集
        x_test = np.vstack((outliers, x_test))
        y_test = np.hstack((outlier_labels, y_test))
        # y_test[y_test == 'o'] = -1; y_test[y_test == 'n'] = 1
        # 把y_test中满足条件 y_test == 'o'的输出为-1，其他输出为1。最后，转化为list，方便计算roc
        y_test = list(np.where(y_test == 'o', -1, 1))

        for train_size in np.arange(0.88, 1, 0.01):
            train_size = round(train_size, 2)
            # 划分其他为训练集和验证集
            x_train, x_cv, y_train, y_cv = train_test_split(x, y, train_size=train_size, random_state=3)

            # print('-' * 30)
            # print("outer train size:", outer_train_size)
            # print("train size:", train_size)
            multi_model(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, threshold=0.5)
            # print('-' * 30)
