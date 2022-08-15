#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Jinzhong Xu
# @Contact : jinzhongxu@csu.ac.cn
# @Time    : 2021/11/12 14:55
# @File    : now.py
# @Software: PyCharm

import time


def current_time(s='%Y-%m-%d %H:%M:%S'):
    return time.strftime(s, time.localtime(time.time()))
