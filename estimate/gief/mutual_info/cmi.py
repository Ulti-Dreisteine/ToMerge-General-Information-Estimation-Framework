# -*- coding: utf-8 -*-
"""
Created on 2022/09/18 16:32:51

@File -> cmi.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 条件互信息
"""

import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
sys.path.insert(0, BASE_DIR)

from estimate.setting import DTYPES, BASE
from estimate.util import stdize_values
from ._kraskov import kraskov_mi as cal_kraskov_cmi
from .mi import MutualInfoGIEF
from estimate.gief.entropy.cond_entropy import CondEntropy
from estimate.gief.entropy.marg_entropy import MargEntropy


class CondMutualInfoGIEF(object):
    """条件互信息"""
    
    def __init__(self, x: np.ndarray, xtype: str, y: np.ndarray, ytype: str, z: np.ndarray, ztype: str):
        assert xtype in DTYPES
        assert ytype in DTYPES
        assert ztype in DTYPES
        self.x = x.copy()
        self.y = y.copy()
        self.z = z.copy()
        self.xtype = xtype
        self.ytype = ytype
        self.ztype = ztype
        
    def __call__(self, k=3, metric="chebyshev", method: str=None):
        # 如果指定使用Kraskov方法, 或者三个变量数值类型均为"c", 则返回Kraskov熵
        cond_0 = method=="Kraskov"
        cond_1 = (self.xtype=="c") & (self.ytype=="c") & (self.ztype=="c")
        if cond_0 | cond_1:
            x, y, z = stdize_values(self.x, self.xtype), stdize_values(self.y, self.ytype), \
                stdize_values(self.z, self.ztype)
            return cal_kraskov_cmi(x, y, z, k=k, base=BASE, alpha=0)
        
        # 根据z与x或y类型差异选择计算方法
        if self.ztype in [self.xtype, self.ytype]:
            if self.ztype == self.xtype:
                x2, x1 = self.x, self.y
                x2_type, x1_type = self.xtype, self.ytype
            else:
                x2, x1 = self.y, self.x
                x2_type, x1_type = self.ytype, self.xtype
            
            arr = np.c_[x2, self.z]
            return MutualInfoGIEF(arr, x2_type, x1, x1_type)(k, metric) - \
                MutualInfoGIEF(x1, x1_type, self.z, self.ztype)(k, metric)
        else:
            a1 = MutualInfoGIEF(self.x, self.xtype, self.y, self.ytype)(k, metric)
            a2 = CondEntropy(self.z, self.ztype, self.x, self.xtype)(k, metric)
            a3 = CondEntropy(self.z, self.ztype, self.y, self.ytype)(k, metric)
            
            arr = np.c_[self.x, self.y]
            a4 = CondEntropy(self.z, self.ztype, arr, self.xtype)(k, metric)
            a5 = MargEntropy(self.z, self.ztype)(k, metric)
            return a1 + a2 + a3 - a4 - a5