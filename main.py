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
from tqdm import tqdm, trange
from tda.timestamps import display_time
from multiprocessing import Pool
import warnings

warnings.filterwarnings("ignore")


def main(cluster):
    # 读取数据
    path = './data'
    filenames = os.listdir(path=path)
    data_paths = [os.path.join(path, file) for file in filenames]
    # 模型比较
    for file_path in data_paths:
        run.just_do_it(path=file_path, cluster=cluster, multiple=2, random_state=3)


if __name__ == '__main__':
    clusters = ['tomato', 'spectral', 'hierarchical', 'dbscan', 'optics', 'birch', 'kmeans']

    # # 单核计算
    # clusters = ['tomato']
    # for cluster in clusters:
    #     print("使用聚类方法 {0} 进行处理 ...".format(cluster))
    #     main(cluster=cluster)
    #     print("\t聚类方法 {0} 处理结束 ^_^".format(cluster))
    # print("整个任务完成！")

    # 使用CPU 多核心多进程加速计算

    with Pool(5) as p:
        p.map(main, clusters)
    print("整个任务完成！")
