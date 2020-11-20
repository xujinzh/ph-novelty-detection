#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Jinzhong Xu
# @Contact : jinzhongxu@csu.ac.cn
# @Time    : 2020/8/21 18:27
# @File    : main.py
# @Software: PyCharm

from tda import run


def main():
    # 读取数据
    data_path = "./data/breast-cancer-unsupervised-ad.csv"
    # data_path = "./data/satellite-unsupervised-ad.csv"
    # data_path = "./data/pen-global-unsupervised-ad.csv"
    # data_path = './data/annthyroid-unsupervised-ad.csv'
    run.just_do_it(path=data_path)


if __name__ == '__main__':
    main()
    print('success')
