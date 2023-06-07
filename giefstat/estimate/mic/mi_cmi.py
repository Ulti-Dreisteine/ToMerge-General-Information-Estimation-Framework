# -*- coding: utf-8 -*-
"""
Created on 2022/09/18 23:21:37

@File -> mi_cmi.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: MIC和RMIC
"""

import numpy as np

from ...util import SuperCategorEncoding, discretize_arr
from ._mic_rmic import MaximalInfoCoeff, RefinedMaximalInfoCoeff


# ---- 数据压缩编码 ---------------------------------------------------------------------------------

def _transf_labels2char(s: np.ndarray):
    """哎, numpy直接匹配函数没找到, 只能暂时转成字符串来处理了"""
    return "".join(f"-{int(p)}" for p in s)


def _convert_arr2int(arr: np.ndarray):
    """将多维数据按照标签进行编码为一维整数
    :param arr: 具有离散整数值的多维数组
    """
    combines = [tuple(row) for row in arr]
    labels = np.unique(combines, axis=0)
    labels = np.apply_along_axis(_transf_labels2char, 1, labels)
    arr_labels = np.apply_along_axis(_transf_labels2char, 1, arr).reshape(-1, 1)
    return np.apply_along_axis(lambda s: np.where(labels == s)[0][0], 1, arr_labels)


def _reencode(arr: np.ndarray):
    """重新将多维数组编码为1维数组"""
    arr = arr.copy()
    arr = discretize_arr(arr, n = arr.shape[0] // 10, method="qcut")
    arr = _convert_arr2int(arr)
    return arr


class MIC(object):
    """最大信息系数"""
    
    def __init__(self, x, y):
        self.x = x.copy().reshape(len(x), -1)
        self.y = y.copy().reshape(len(y), -1)
    
    def __call__(self, method="rmic", encode=True):
        # 多维数组逐列离散化, 并合并编码压缩为一维数组
        y = _reencode(self.y) if self.y.shape[1] > 1 else self.y.copy()
        x = _reencode(self.x) if self.x.shape[1] > 1 else self.x.copy()
        if method != "rmic":
            return MaximalInfoCoeff(x, y).cal_assoc()
        
        if encode:
            x = SuperCategorEncoding(x, y).encode(method="mhg") # 进行有监督编码, NOTE x值需为int
            
        return RefinedMaximalInfoCoeff(x, y).cal_assoc()
            
            
class CMIC(object):
    """条件最大信息系数"""
    
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        
    def __call__(self, method="rmic"):
        xy = np.c_[self.x, self.y]
        return MIC(self.x, self.y)(method) + MIC(xy, self.z)(method) - MIC(self.x, self.z)(method) - \
            MIC(self.y, self.z)(method)