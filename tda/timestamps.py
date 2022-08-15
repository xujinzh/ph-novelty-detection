# 装饰器用来计算函数运行时间
import time


def display_time(text):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            print(f'{now} {text} time consumed: {end_time - start_time} seconds')
            return result

        return wrapper

    return decorator
