import os
import pandas as pd
import re
import numpy as np
from pathlib import Path, PurePath
from statistics import mean


def compare(output1, output2):
    """
    对比不同策略下，各结果的差异
    """
    output1 = os.path.expanduser(output1)
    output2 = os.path.expanduser(output2)
    csv_files_output1 = [os.path.join(output1, file) for file in os.listdir(output1) \
                         if os.path.splitext(file)[1] == '.csv']
    csv_files_output2 = [os.path.join(output2, file) for file in os.listdir(output2) \
                         if os.path.splitext(file)[1] == '.csv']
    d = {'datasets': [], 'cluster': [], 'optim': [], 'lof': [], 'svm': [], \
         'ph1_max': [], 'ph2_max': [], 'ph1_max >= ph2_max': [], \
         'ph1_mean': [], 'ph2_mean': [], 'ph1_mean >= ph2_mean': []}
    for csv_file1 in csv_files_output1:
        path1, csv1 = os.path.split(csv_file1)
        for csv_file2 in csv_files_output2:
            path2, csv2 = os.path.split(csv_file2)
            if csv1 == csv2:
                l = re.split(r'(\W+)', csv1)
                df1 = pd.read_csv(csv_file1)
                df2 = pd.read_csv(csv_file2)
                ph1 = df1.loc[:, 'ph']
                ph2 = df2.loc[:, 'ph']
                lof = df1.loc[:, 'lof'][0]
                svm = df1.loc[:, 'svm'][0]
                ph1_max = ph1[0]
                ph2_max = ph2[0]
                ph1_mean = np.mean(ph1)
                ph2_mean = np.mean(ph2)
                d['datasets'].append(l[-7])
                d['cluster'].append(l[-3])
                d['lof'].append(lof)
                d['svm'].append(svm)
                d['ph1_max'].append(ph1_max)
                d['ph2_max'].append(ph2_max)
                values = [lof, svm, ph1_max, ph2_max]
                values_max = max(values)
                optim_algo = []
                for algo, value in zip(['lof', 'svm', 'ph1', 'ph2'], values):
                    if value == values_max:
                        optim_algo.append(algo)
                d['optim'].append(optim_algo)
                d['ph1_max >= ph2_max'].append(ph1_max >= ph2_max)
                d['ph1_mean'].append(ph1_mean)
                d['ph2_mean'].append(ph2_mean)
                d['ph1_mean >= ph2_mean'].append(ph1_mean >= ph2_mean)
    df_res = pd.DataFrame(data=d)
    return df_res.sort_values(by=['datasets', 'cluster'])


def compare_triple(output1, output2, output3):
    # 获取CSV文件
    output1_csv = list(Path(output1).glob("*.csv"))
    output2_csv = list(Path(output2).glob("*.csv"))
    output3_csv = list(Path(output3).glob("*.csv"))

    output1_csv_filename = [file.name for file in output1_csv]
    output2_csv_filename = [file.name for file in output2_csv]
    output3_csv_filename = [file.name for file in output3_csv]
    # 获取三个文件夹中共同含有的CSV文件名
    common_csv = set(output1_csv_filename) & set(output2_csv_filename) & set(output3_csv_filename)

    d_mean_info = {
        "data": [],
        "cluster": [],
        "F": [],
        "S": [],
        "LOF": [],
        "SVM": [],
        "PH": [],
    }  # 记录计算得到的最大值的平均值结果，包含数据、聚类算法、PH、LOF、SVM
    if not ((len(output1_csv) == len(output2_csv)) and (len(output2_csv) == len(output3_csv))):
        print("三个数据个数不一样")
    for file1 in output1_csv:  # 针对第一个数据集进行遍历，三个数据集数量要求一样
        if file1.name not in common_csv: # 如果不是三者都有的CSV，则跳过
            continue
        name = file1.name  # 获取文件名，不包含文件夹路径
        name_list = re.split(r"(\W+)", name)  # 获取数据名、算法名等信息；W 匹配特殊字符，即非字母、非数字、非汉字、非_
        file2 = PurePath(output2) / name  # 获取第二个数据结果的完整路径
        file3 = PurePath(output3) / name  # 获取第三个数据结果的完整路径
        # 分别读取三个数据结果
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        df3 = pd.read_csv(file3)
        # 把LOF的最大值分别读取出来，计算平均值
        lof_max_file1 = df1.loc[0, "lof"]
        lof_max_file2 = df2.loc[0, "lof"]
        lof_max_file3 = df3.loc[0, "lof"]
        lof_mean = mean([lof_max_file1, lof_max_file2, lof_max_file3])
        # 把SVM的最大值分别读取出来，计算平均值
        svm_max_file1 = df1.loc[0, "svm"]
        svm_max_file2 = df2.loc[0, "svm"]
        svm_max_file3 = df3.loc[0, "svm"]
        svm_mean = mean([svm_max_file1, svm_max_file2, svm_max_file3])
        # 把PH的最大值分别读取出来，计算平均值
        ph_max_file1 = df1.loc[0, "ph"]
        ph_max_file2 = df2.loc[0, "ph"]
        ph_max_file3 = df3.loc[0, "ph"]
        ph_mean = mean([ph_max_file1, ph_max_file2, ph_max_file3])
        # 获取最大、第二大的值对应的算法
        value_set = set([lof_mean, svm_mean, ph_mean])
        max_mean = max(value_set)
        f_max = []
        if lof_mean == max_mean:
            f_max.append("LOF")
        if svm_mean == max_mean:
            f_max.append("SVM")
        if ph_mean == max_mean:
            f_max.append("PH")
        d_mean_info["F"].append(f_max)
        # 记录第二大
        value_without_max_set = value_set - set([max_mean])
        if len(value_without_max_set) == 0:  # 如果集合中所有元素都相等，都是最大值
            d_mean_info["S"].append(["LOF", "SVM", "PH"])
        else:
            max_second_mean = max(value_without_max_set)
            s_max = []
            if lof_mean == max_second_mean:
                s_max.append("LOF")
            if svm_mean == max_second_mean:
                s_max.append("SVM")
            if ph_mean == max_second_mean:
                s_max.append("PH")
            d_mean_info["S"].append(s_max)
        # 记录其他信息，包括数据集、聚类算法、LOF的平均值、SVM的平均值、PH的平均值
        d_mean_info["data"].append(name_list[2])
        d_mean_info["cluster"].append(name_list[6])
        d_mean_info["LOF"].append(lof_mean)
        d_mean_info["SVM"].append(svm_mean)
        d_mean_info["PH"].append(ph_mean)
    ddf = pd.DataFrame(d_mean_info)

    return ddf.sort_values(by=["data", "cluster"])
