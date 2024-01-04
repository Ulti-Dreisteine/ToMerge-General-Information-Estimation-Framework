# -*- coding: utf-8 -*-
"""
Created on 2023/01/05 20:09:48

@File -> kde.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 基于KDE的信息熵和互信息估计, 仅适用于连续变量
"""

from scipy.stats import gaussian_kde
from numpy import log
import numpy as np

from ...setting import BASE
from ...util import stdize_values


# ---- 数据信息估计 ---------------------------------------------------------------------------------

class MargEntropy(object):
    """边际熵"""
    # NOTE: 仅适用于连续变量
    
    def __init__(self, x: np.ndarray):
        self.x_norm = stdize_values(x, "c")  # shape = (N, dim)
        assert self.x_norm is not None
        self.dim = self.x_norm.shape[1]
        
    def __call__(self):
        dens = gaussian_kde(self.x_norm.T)(self.x_norm.T)  # shape = (N,)
        return np.mean(-log(dens) / log(BASE))
    
    
class MutualInfoKDE(object):
    """互信息"""
    # NOTE: 仅适用于连续变量
    
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x_norm = stdize_values(x, "c")  # shape = (N, dim)
        self.y_norm = stdize_values(y, "c")
        self.xy_norm = np.c_[self.x_norm, self.y_norm]
        assert self.x_norm is not None
        assert self.y_norm is not None
        self.dim_x = self.x_norm.shape[1]
        self.dim_y = self.y_norm.shape[1]
        
    def __call__(self):
        dens_xy = gaussian_kde(self.xy_norm.T)(self.xy_norm.T)  # shape = (N,)
        dens_x = gaussian_kde(self.x_norm.T)(self.x_norm.T)  # shape = (N,)
        dens_y = gaussian_kde(self.y_norm.T)(self.y_norm.T)  # shape = (N,)
        return np.mean(log(dens_xy / (dens_x * dens_y)) / log(BASE))