import gudhi as gh
import numpy as np
import pandas as pd
from sklearn import preprocessing
from tqdm import tqdm
import copy
from TimeStamps import display_time


class PHNovDet(object):

    def __init__(self, data=None, max_edge_length=42, max_dimension=1, homology_coeff_field=2, min_persistence=0,
                 threshold=0.05, base=20):
        self.data = data
        self.max_edge_length = max_edge_length
        self.max_dimension = max_dimension
        self.homology_coeff_field = homology_coeff_field
        self.min_persistence = min_persistence
        self.threshold = threshold
        self.base = base
        self.scores = np.array([])

    def ph(self, plot=True):

        #         print("RipsComplex creation from points")
        data = preprocessing.scale(self.data)
        rips = gh.RipsComplex(points=data, max_edge_length=self.max_edge_length)
        simplex_tree = rips.create_simplex_tree(max_dimension=self.max_dimension)
        diag = simplex_tree.persistence(homology_coeff_field=self.homology_coeff_field,
                                        min_persistence=self.min_persistence)
        #         print("diag=", diag)

        if plot == True:
            pplot = gh.plot_persistence_diagram(diag)
            pplot.show()
        return diag

    def bottleneck(self, point, plot=True):
        diag = self.ph(plot)
        diag0 = []
        for i, d in enumerate(diag):
            diag0.append(list(diag[i][1]))

        temp_data = self.data
        self.data = np.vstack((self.data, point))
        diag_point = self.ph(plot)
        diag_point0 = []
        for i, d in enumerate(diag_point):
            diag_point0.append(list(diag_point[i][1]))

        self.data = temp_data

        return gh.bottleneck_distance(diag0, diag_point0)

    @display_time
    def grow(self, input_path, outlier_number, output_path, threshold=1.0, base_index=10, tree_height=100):
        '''
        param input_path: pandas数据集的路径
        param outlier_number: 数据集中包含的异常点数
        param output_number: 找到的形状数据集B的输出路径
        param threshold: 判断是否保留随机选中的数据点的阈值，大于该阈值丢弃该点，小于该阈值保留该点
        param base_index: 期初形状数据集B的点的个数
        param tree_height: 生长形状数据集B的最大个数
        '''
        satellite = pd.read_csv(input_path, header=None)
        satellite_list = list(np.arange(outlier_number, len(satellite.index)))
        if base_index > len(satellite_list):
            print('out of bound about base_index')

        else:
            base_list = list(np.random.choice(satellite_list, base_index, replace=False))
            self.data = np.array(satellite.iloc[base_list, :-1])
            for i in tqdm(np.arange(outlier_number, len(satellite.index))):
                # 从第85个数据开始验证并生长形状
                point = np.array(satellite.iloc[i, :-1])  # 索引出每一个待验证的数据点

                if i in base_list:
                    continue
                else:
                    if len(self.data) < tree_height:
                        if self.bottleneck(point, plot=False) <= threshold:
                            # 以阈值1为验证条件，小于1的添加到B数据中，大于1的舍弃
                            base_list.append(i)  # 将生长的数据索引取出
                            self.data = np.vstack((self.data, point))  # 将通过验证的数据进行生长
                    else:
                        break

            B = satellite.iloc[base_list, :]  # 生成的形状数据B
            B.to_csv(output_path)  # 将长成的形状数据保存1起来
            #             print('base_list: ', base_list)
            print('self data shape: ', self.data.shape[0])  # 打印计算中使用的生长数据的形状
            print('B data shape: ', B.shape)  # 打印出真实生长数据的形状

    @display_time
    def crop(self, input_path, outlier_number, output_path, threshold=1.0, base_lower=10, cycle=100):
        '''
        param input_path: pandas数据集的路径
        param outlier_number: 数据集中包含的异常点数
        param output_number: 找到的形状数据集B的输出路径
        param threshold: 判断是否保留随机选中的数据点的阈值，大于该阈值丢弃该点，小于该阈值保留该点
        param base_lower: 形状数据集B的最小点数
        param cycle: 随机筛选数据集的次数
        '''
        satellite = pd.read_csv(input_path, header=None)
        self.data = np.array(satellite.iloc[outlier_number:, :-1])
        base_list = list(np.arange(outlier_number, len(satellite.index)))
        if base_lower > len(self.data):  # 如果基数集的个数设置的过大，超过A的点数，则返回
            print('base lower is out of bound')
        else:  # 如果不超过A的点数，则继续
            for i in tqdm(np.arange(cycle)):  # 循环随机从现有的数据集索引中筛选数据点
                choice_index = np.random.choice(base_list, 1)  # 随机筛选一个点，以相同概率
                point = np.array(satellite.iloc[choice_index, :-1])  # 找到该索引对应的数据点
                temp_list = copy.deepcopy(base_list)  # 保留现在的索引列表，以防上面筛选的点没有被丢弃
                base_list.remove(choice_index)  # 暂时丢弃筛选的点，看看是否导致形状改变很大
                self.data = np.array(satellite.iloc[base_list, :-1])  # 给出剔除筛选数据点后的数据集
                if len(self.data) < base_lower:  # 如果提出后基数集B个数小于设定的界限，则停止循环，直接返回
                    break
                else:  # 没有低于最小界限，则继续
                    if self.bottleneck(point, plot=False) <= threshold:
                        # 如果选中的点能够有效代表整个数据集的形状，则保留
                        base_list = copy.deepcopy(temp_list)
                        self.data = np.array(satellite.iloc[base_list, :-1])
                    else:  # 如果不能，则剔除该点
                        #                         base_list = copy.deepcopy(base_list)
                        #                         self.data = np.array(satellite.iloc[base_list, :-1])
                        continue

            B = satellite.iloc[base_list, :]  # 生成的形状数据B
            B.to_csv(output_path)  # 将长成的形状数据保存1起来
            #             print('base_list: ', base_list)
            print('self data shape: ', self.data.shape[0])  # 打印计算中使用的生长数据的形状
            print('B data shape: ', B.shape)  # 打印出真实生长数据的形状

    @display_time
    def reduce(self, input_path, outlier_number, output_path, threshold=1.0, base_lower=10):
        '''
        param input_path: pandas数据集的路径
        param outlier_number: 数据集中包含的异常点数
        param output_number: 找到的形状数据集B的输出路径
        param threshold: 判断是否保留随机选中的数据点的阈值，大于该阈值丢弃该点，小于该阈值保留该点
        param base_lower: 形状数据集B的最小点数
        '''
        satellite = pd.read_csv(input_path, header=None)
        self.data = np.array(satellite.iloc[outlier_number:, :-1])
        base_list = list(np.arange(outlier_number, len(satellite.index)))
        if base_lower > len(self.data):  # 如果基数集的个数设置的过大，超过A的点数，则返回
            print('base lower is out of bound')
        else:  # 如果不超过A的点数，则继续
            for i in tqdm(np.arange(outlier_number, len(satellite.index))):  # 循环随机从现有的数据集索引中筛选数据点
                choice_index = i  # 从前往后，一个一个排查
                point = np.array(satellite.iloc[choice_index, :-1])  # 找到该索引对应的数据点
                temp_list = copy.deepcopy(base_list)  # 保留现在的索引列表，以防上面筛选的点没有被丢弃
                base_list.remove(choice_index)  # 暂时丢弃筛选的点，看看是否导致形状改变很大
                self.data = np.array(satellite.iloc[base_list, :-1])  # 给出剔除筛选数据点后的数据集
                if len(self.data) < base_lower:  # 如果提出后基数集B个数小于设定的界限，则停止循环，直接返回
                    break
                else:  # 没有低于最小界限，则继续
                    if self.bottleneck(point, plot=False) <= threshold:
                        # 如果选中的点能够有效代表整个数据集的形状，则保留
                        base_list = copy.deepcopy(temp_list)
                    #                         self.data = np.array(satellite.iloc[base_list, :-1])
                    else:  # 如果不能，则剔除该点
                        #                         base_list = base_list
                        #                         self.data = np.array(satellite.iloc[base_list, :-1])
                        continue

            B = satellite.iloc[base_list, :]  # 生成的形状数据B
            B.to_csv(output_path)  # 将长成的形状数据保存1起来
            #             print('base_list: ', base_list)
            print('self data shape: ', self.data.shape[0])  # 打印计算中使用的生长数据的形状
            print('B data shape: ', B.shape)  # 打印出真实生长数据的形状

    @display_time
    def fit(self, x_train, output_path='.'):
        '''
        param input_path: pandas数据集的路径
        param outlier_number: 数据集中包含的异常点数
        param output_number: 找到的形状数据集B的输出路径
        param threshold: 判断是否保留随机选中的数据点的阈值，大于该阈值丢弃该点，小于该阈值保留该点
        param base_lower: 形状数据集B的最小点数
        '''
        self.data = x_train
        data = pd.DataFrame(data=x_train)
        base_list = list(data.index)
        abandom_point = []

        if self.base > len(data):
            print("基础数据集的势太小")
        else:
            try:
                with tqdm(np.arange(len(data.index))) as t:
                    for i in t:
                        choice_index = i
                        choice_point = np.array(data.iloc[choice_index, :])
                        temp_list = copy.deepcopy(base_list)
                        base_list.remove(choice_index)
                        self.data = np.array(data.iloc[base_list, :])
                        if len(self.data) < self.base:
                            break
                        else:
                            if self.bottleneck(choice_point, plot=False) < self.threshold:
                                base_list = copy.deepcopy(temp_list)
                            else:
                                abandom_point.append(choice_index)
                                continue
            except KeyboardInterrupt:
                t.close()
                raise

            t.close()

            B = data.iloc[base_list, :]
            self.data = np.array(B)
            B.to_csv(output_path + ".csv")
            print("self data shape: ", data.shape[0])
            print("B data shape: ", B.shape)
            with open(output_path + ".txt", 'w') as f:
                for item in abandom_point:
                    f.write("%s\n" % item)

    @display_time
    def predict(self, x_test):
        y_scores = []
        for i in range(len(x_test)):
            point = x_test[i]
            y_scores.append(self.bottleneck(point, plot=False))
        y_scores = np.array(y_scores)
        self.scores = copy.deepcopy(y_scores)
        y_scores[y_scores >= self.threshold] = 1
        y_scores[y_scores < self.threshold] = -1
        return -y_scores

    @display_time
    def score_samples(self, x_test):
        self.predict(x_test)
        return self.scores

    @display_time
    def score_samples(self):
        return self.scores
