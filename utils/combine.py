#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Jinzhong Xu
# @Contact : jinzhongxu@csu.ac.cn
# @Time    : 2021/9/23 10:51
# @File    : combine.py
# @Software: PyCharm


import itertools
import copy
import numpy as np


def combine(*args):
    """
    把多个列表的元素依次组合，返回一个列表
    """
    Args = list(args)
    num = len(Args)
    # 对于不是列表的输入，转化为单元素列表
    for i in range(num):
        if not isinstance(Args[i], list):
            # print(f"第{i}个参数不是列表，请检查")
            Args[i] = list((Args[i],))
    # 用于合并多个列表
    a0 = copy.deepcopy(Args[0])
    for b in Args[1:]:
        a0.extend(b)
    # 得到所有组合的元组
    res = list(itertools.combinations(a0, num))
    # 用于获取最后的组合
    res0 = copy.deepcopy(res)
    for r in res:
        for i, b in enumerate(Args):
            if r[i] not in b:
                try:
                    res0.remove(r)
                except ValueError:
                    continue
    return res0


if __name__ == '__main__':
    print(combine('tomato', list(range(1, 20)), list(np.arange(0.2, 0.8, 0.1))))
