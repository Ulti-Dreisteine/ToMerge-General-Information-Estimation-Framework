# -*- coding: utf-8 -*-
"""
Created on 2023/04/26 15:24:07

@File -> bootstrap.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 通过自举法获得关联度量值分布
"""

import numpy as np
import random

from .. import cal_general_assoc


def cal_bootstrap_coeff(x, y, method, z=None, xtype=None, ytype=None, ztype=None, rounds=100, 
                        **kwargs):
    """基于自举重采样的关联系数估计

    :param x: x数据, np.ndarray
    :param y: y数据, np.ndarray
    :param method: 采用的度量方法, 参见estimate.__init__, str
    :param z: z数据, np.ndarray, defaults to None
    :param xtype: x的值类型, 参见estimate.__init__, str, defaults to None
    :param ytype: y的值类型, 参见estimate.__init__, str, defaults to None
    :param ztype: z的值类型, 参见estimate.__init__, str, defaults to None
    :param rounds: Bootstrap轮数, int, defaults to 100
    :return: assoc_mean: 均值, assocs_bt: 随机测试后获得的所有关联值结果
    """    
    assocs_bt = np.array([])
    idxs = np.arange(len(x))
    size_bt = len(x)  # NOTE: 这里样本量与原数据一致, 以消除样本量变化对估计结果的影响
    
    for _ in range(rounds):
        idxs_bt = random.choices(idxs, k=size_bt)  # NOTE: 有放回抽样
        x_bt, y_bt = x[idxs_bt], y[idxs_bt]
        z_bt = z[idxs_bt] if z is not None else z
        assoc_ = cal_general_assoc(x_bt, y_bt, z_bt, method, xtype, ytype, ztype, **kwargs)
        
        assocs_bt = np.append(assocs_bt, assoc_)
        assoc_mean = np.mean(assocs_bt)
    
    return assoc_mean, assocs_bt