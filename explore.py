#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Jinzhong Xu
# @Contact : jinzhongxu@csu.ac.cn
# @Time    : 2020/8/21 18:27
# @File    : explore.py
# @Software: PyCharm

import argparse
import os
import warnings
from multiprocessing import Pool

from tda import train
from tda.timestamps import display_time

warnings.simplefilter("ignore")


def explore(cluster):
    '''
    对最优结果进行探索
    Parameters
    ----------
    cluster：聚类算法

    Returns
    -------

    '''
    # 配置输入参数
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--data', required=False, type=str, default='data', help='data path')
    ap.add_argument('-m', '--multiple', required=False, type=float, default=1.0,
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
        csv_file = os.path.basename(file_path)
        dataname = csv_file.split('-')[0]
        output_csv_name = f'data={dataname}-cluster={cluster}.csv'
        if output_csv_name in os.listdir('output'):
            # 如果上次已经运行出结果，则跳过，不重复运行
            continue
        train.just_do_it(path=file_path, cluster=cluster, multiple=multiple, random_state=3)


@display_time("all")
def do():
    clusters = ['tomato', 'spectral', 'hierarchical', 'birch', 'kmeans', 'optics', 'dbscan']

    # 使用CPU 多核心多进程加速计算
    p = Pool(processes=max(os.cpu_count() - 1, 1))
    p.map(explore, clusters)

    # # 使用单进程计算
    # for cluster in clusters:
    #     explore(cluster)

    print("整个任务完成！")


if __name__ == '__main__':
    do()
