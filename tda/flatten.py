#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Jinzhong Xu
# @Contact : jinzhongxu@csu.ac.cn
# @Time    : 2021/9/15 18:41
# @File    : flatten.py
# @Software: PyCharm


from collections import Counter


def small_into_big(labels, small_value, big_value):
    """
    把一个列表labels中指定的元素small_value，用一个值big_value替换
    """
    labels = [big_value if value == small_value else value for value in labels]
    return labels


def flatten_ones(labels):
    """
    把列表中的元素按照出现的次数摊平，使得元素出现次数趋近均等
    具体地，将出现最小的元素用第二小的元素替换，如果两者都小于平均个数
    :param labels: 列表
    :return: 更改后的列表
    """
    # 统计每个元素出现的次数
    counter_labels = Counter(labels)
    # 如果聚类中心数小于 centroid，则不摊平
    if len(counter_labels) <= 1:
        return labels
    # 按照出现的次数从大到小排序
    sort_labels = counter_labels.most_common()
    # 合并第一少到第二少的标签， 递归执行检查和替换
    labels = small_into_big(labels=labels, small_value=sort_labels[-1][0],
                            big_value=sort_labels[-2][0])
    return labels


def flatten(labels, centroid=100):
    """
    把列表labels中的元素按照出现的次数，最少的元素更改为第二少的元素，
    知道类别数小于centroid
    :param labels: 列表
    :param centroid: 类别个数
    :return: 元素类别个数等于centroid的列表
    """
    while len(set(labels)) > centroid:
        labels = flatten_ones(labels)
    return labels
