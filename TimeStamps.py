# 装饰器用来计算函数运行时间

import time


def display_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print('Total time: {:.4} seconds'.format(end_time - start_time))
        return result

    return wrapper
