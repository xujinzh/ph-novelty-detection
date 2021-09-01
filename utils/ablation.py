import os
import pandas as pd
import re
import numpy as np

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
    d = {'datasets':[], 'cluster':[], 'optim':[], 'lof':[], 'svm':[], \
       'ph1_max':[], 'ph2_max':[], 'ph1_max >= ph2_max':[], \
       'ph1_mean':[], 'ph2_mean':[], 'ph1_mean >= ph2_mean':[]}
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
