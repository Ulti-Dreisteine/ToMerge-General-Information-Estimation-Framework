# -*- coding: utf-8 -*-
# sourcery skip: avoid-builtin-shadow
"""
Created on 2022/09/18 15:28:09

@File -> marg_entropy.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 边际熵
"""

__doc__ = """
    本代码用于对一维和多维离散或连续变量数据的信息熵和互信息进行计算.
    连续变量信息熵使用Kraskov等人的方法计算, Lord等人文献可作为入门;离散变量信息熵则直接进行计算.

    求得信息熵和互信息后, 便可对条件熵进行求取:

    H(Y|X) = H(Y) - I(X;Y)

    此处不再补充条件熵的相关代码.

    参考文献
    1. W. M. Lord, J. Sun, E. M. Bolt: Geometric K-Nearest Neighbor Estimation of Entropy and Mutual information, 2018.
    2. A. Kraskov, H. Stoegbauer, P. Grassberger: Estimating Mutual Information, 2003.
    3. D. Lombardi, S. Pant: A Non-Parametric K-Nearest Neighbor Entropy Estimator, 2015.
    4. B. C. Ross: Mutual Information between Discrete and Continuous Data Sets, 2014.
"""

from scipy.special import psi
from numpy import log
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 4))
sys.path.insert(0, BASE_DIR)

from estimate.setting import DTYPES, BASE
from estimate.util import stdize_values, build_tree, query_neighbors_dist, get_unit_ball_volume


def _cal_discrete_entropy(x):
    _, counts = np.unique(x, return_counts=True, axis=0)
    proba = counts.astype(float) / len(x)
    proba = proba[proba > 0.0]
    return np.sum(proba * np.log(1. / proba))


def _cal_kl_entropy(x, k, metric="chebyshev"):
    assert k <= len(x) - 1
    N, D = x.shape
    
    # 构建距离树.
    tree = build_tree(x, "chebyshev")

    # 计算结果.
    # 参见Kraskov, et al. Estimating Mutual Information, 2004. Eq. (20).
    nn_distc = query_neighbors_dist(tree, x, k)  # 获得了各样本第k近邻的距离
    v = get_unit_ball_volume(D, metric)
    return (-psi(k) + psi(N) + np.log(v) + D * np.log(nn_distc).mean())
    

class MargEntropy(object):
    """计算任意连续和离散变量的信息熵"""
    
    def __init__(self, x: np.ndarray, xtype: str):
        assert xtype in DTYPES
        self.x_norm = stdize_values(x, xtype)
        self.xtype = xtype
        
    def __call__(self, k: int=3, metric: str="chebyshev"):
        if self.xtype == "d":
            return _cal_discrete_entropy(self.x_norm) / log(BASE)
        elif self.xtype == "c":
            return _cal_kl_entropy(self.x_norm, k, metric) / log(BASE)
        
        
if __name__ == "__main__":
    def test_c():
        x = np.random.normal(0, 1, 10000)
        y = np.random.normal(0, 1, 10000)
        
        self = MargEntropy(x, "c")
        Hx = self()
        
        self = MargEntropy(y, "c")
        Hy = self()
        
        self = MargEntropy(np.c_[x, y], "c")
        Hxy = self()
        
        print(f"Hx: {Hx} \tHy: {Hy} \tHxy: {Hxy} \tMI: {Hx + Hy - Hxy}") # note 此处MI计算不准确
        
    def test_d():
        x = np.random.randint(0, 5, 10000)
        y = np.random.randint(10, 20, 10000)
        
        self = MargEntropy(x, "d")
        Hx = self()
        
        self = MargEntropy(y, "d")
        Hy = self()
        
        self = MargEntropy(np.c_[x, y], "d")
        Hxy = self()
        
        print(f"Hx: {Hx} \tHy: {Hy} \tHxy: {Hxy} \tMI: {Hx + Hy - Hxy}")  
    
    print("test c")    
    test_c()
    
    print("test d")    
    test_d()