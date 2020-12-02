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
import argparse
import warnings

warnings.simplefilter("ignore")


def main(cluster):
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--data', required=True, type=str, default='./data', help='data path')
    ap.add_argument('-m', '--multiple', required=True, type=int, default=1,
                    help='normal point than novelty point in test data')
    args = vars(ap.parse_args())

    # 读取数据
    path = args['data']
    multiple = args['multiple']
    filenames = os.listdir(path=path)
    data_paths = [os.path.join(path, file) for file in filenames]

    # 模型比较
    for file_path in data_paths:
        if not os.path.isfile(file_path):
            print(f'文件：{file_path} 不是数据集，将跳过该文件！！！')
            continue
        run.just_do_it(path=file_path, cluster=cluster, multiple=multiple, random_state=3)


if __name__ == '__main__':
    clusters = ['tomato', 'spectral', 'hierarchical', 'dbscan', 'optics', 'birch', 'kmeans']

    # 单核计算
    # for cluster in clusters:
    #     print("使用聚类方法 {0} 进行处理 ...".format(cluster))
    #     main(cluster=cluster)
    #     print("\t聚类方法 {0} 处理结束 ^_^".format(cluster))

    # 使用CPU 多核心多进程加速计算
    p = Pool(processes=min(3, os.cpu_count()))
    p.map_async(main, clusters)
    p.close()
    p.join()
    print("整个任务完成！")
