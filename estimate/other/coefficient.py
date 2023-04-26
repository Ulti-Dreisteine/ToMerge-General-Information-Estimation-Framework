# -*- coding: utf-8 -*-
"""
Created on 2022/08/24 22:44:39

@File -> coefficient.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 相关系数计算
    note: 对类别变量进行编码, 然后视作数值变量计算相关系数
"""

from scipy.stats import pearsonr, spearmanr
import numpy as np
import dcor
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 2))
sys.path.insert(0, BASE_DIR)

from _univar_encoding import SuperCategorEncoding
# from mic import PairwiseMIC, PairwiseRMIC


def cal_dist_corr(x, y, x_type: str="c"):
    """距离相关系数"""
    if x_type == "d":
        x = _encode(x, y)
    return np.abs(dcor.distance_correlation(x, y))


def cal_pearson_corr(x, y, x_type: str="c"):
    """Pearson相关系数"""
    if x_type == "d":
        x = _encode(x, y)
    return np.abs(pearsonr(x, y)[0])


def cal_spearman_corr(x, y, x_type: str="c"):
    """Spearman相关系数"""
    if x_type == "d":
        x = _encode(x, y)
    return np.abs(spearmanr(x, y)[0])
    

def _encode(x, y):
    """如果x是类别型变量, 则对x进行编码
    注意: 这里选择有监督的编码,因此入参有y, 其他编码方式可以在univar_encoding里选择
    """
    super_enc = SuperCategorEncoding(x, y)
    return super_enc.mhg_encoding()