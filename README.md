# Persistent Homology for Novelty Detection (PHND)

## 基于拓扑代数同调理论的新颖点检测算法：PHND

### 代码介绍

代码是论文 "Novelty Detection with Topological Signatures" Jinzhong Xu, Junrong Du, Ye Li, Lele Xu, Lili Guo, Xuzhi Li. 的 Python 实现。

### 代码使用方法

1. git clone https://github.com/xujinzh/ph-novelty-detection.git
2. python main.py

数据使用的是 Harvard Dataverse 中的 **Unsupervised Anomaly Detection Benchmark**，更多视频请访问 [[Unsupervised Anomaly Detection Dataverse](https://dataverse.harvard.edu/dataverse/unsupervised-ad) (Kyushu University)](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OPQMVF) 下载。

### 依赖包

- python3
- numpy
- gudhi
- matplotlib
- scikit-learn
- pandas