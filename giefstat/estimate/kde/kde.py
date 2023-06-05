# -*- coding: utf-8 -*-
"""
Created on 2023/01/05 20:09:48

@File -> kde.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 基于KDE的信息熵和互信息估计, 仅适用于连续变量
"""

from sklearn.preprocessing import MinMaxScaler
from scipy.stats import gaussian_kde
from numpy import log
import numpy as np

BASE = np.e


# ---- 数据标准化处理 --------------------------------------------------------------------------------

def normalize(X):
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X.copy())
    return X


def _convert_1d_series2int(x):
    """将一维数据按照标签进行编码为连续整数"""
    x = x.flatten()
    x_unique = np.unique(x)

    if len(x_unique) > 100:
        raise RuntimeWarning(
            f"too many labels: {len(x_unique)} for the discrete data")

    x = np.apply_along_axis(lambda x: np.where(
        x_unique == x)[0][0], 1, x.reshape(-1, 1))
    return x


def _convert_arr2int(arr):
    """将一维数据按照标签进行编码为连续整数"""
    _, D = arr.shape
    for d in range(D):
        arr[:, d] = _convert_1d_series2int(arr[:, d])
    return arr.astype(int)


def stdize_values(x: np.ndarray, dtype: str, eps: float = 1e-6):
    """数据预处理: 标签值整数化、连续值归一化, 将连续和离散变量样本处理为对应的标准格式用于后续分析"""
    x = x.copy()
    x = x.reshape(x.shape[0], -1)
    if dtype == "c":
        # 连续值加入噪音并归一化
        x += eps * np.random.random_sample(x.shape)
        return normalize(x)
    elif dtype == "d":
        # 将标签值转为连续的整数值
        x = _convert_arr2int(x)
        return x


# ---- 数据信息估计 ---------------------------------------------------------------------------------

class MargEntropy(object):
    """边际熵"""
    # NOTE仅适用于连续变量
    
    def __init__(self, x: np.ndarray):
        self.x_norm = stdize_values(x, "c")  # shape = (N, dim)
        self.dim = self.x_norm.shape[1]
        
    def __call__(self):
        dens = gaussian_kde(self.x_norm.T)(self.x_norm.T)  # shape = (N,)
        return np.mean(-log(dens) / log(BASE))
    
    
class MutualInfoKDE(object):
    """互信息"""
    # NOTE仅适用于连续变量
    
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x_norm = stdize_values(x, "c")  # shape = (N, dim)
        self.y_norm = stdize_values(y, "c")
        self.xy_norm = np.c_[self.x_norm, self.y_norm]
        self.dim_x = self.x_norm.shape[1]
        self.dim_y = self.y_norm.shape[1]
        
    def __call__(self):
        dens_xy = gaussian_kde(self.xy_norm.T)(self.xy_norm.T)  # shape = (N,)
        dens_x = gaussian_kde(self.x_norm.T)(self.x_norm.T)  # shape = (N,)
        dens_y = gaussian_kde(self.y_norm.T)(self.y_norm.T)  # shape = (N,)
        return np.mean(log(dens_xy / (dens_x * dens_y)) / log(BASE))