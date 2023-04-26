# -*- coding: utf-8 -*-
"""
Created on 2023/04/26 16:45:20

@File -> surrog_indep_test.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 基于代用数据的独立性测试
"""

import numpy as np

from . import cal_general_assoc


def exec_surrog_indep_test(x, y, method, z=None, xtype=None, ytype=None, ztype=None, rounds=100, 
                           alpha=0.05, **kwargs):
    """执行基于代用数据的独立性检验

    :param x: x数据, np.ndarray
    :param y: y数据, np.ndarray
    :param method: 选用的关联度量方法, str
    :param z: z数据, np.ndarray, defaults to None
    :param xtype: x的数值类型, str, defaults to None
    :param ytype: y的数值类型, str, defaults to None
    :param ztype: y的数值类型, str, defaults to None
    :param rounds: 重复轮数, defaults to 100
    :param alpha: 显著性阈值, defaults to 0.05
    :param kwargs: cal_general_assoc方法中的关键字参数
    :return 
        assoc: 关联值, float
        p: P值, float
        indep: 是否独立, bool
        assocs_srg: 代理数据关联值数组, np.ndarray
    """
    # 计算关联系数
    assoc = cal_general_assoc(x, y, z, method, xtype, ytype, ztype, **kwargs)
    
    # 计算背景值
    idxs = np.arange(len(x))
    assocs_srg = np.array([])
    for _ in range(rounds):
        idxs_srg = np.random.permutation(idxs)
        x_srg = x[idxs_srg]
        assoc_srg = cal_general_assoc(x_srg, y, z, method, xtype, ytype, ztype, **kwargs)
        assocs_srg = np.append(assocs_srg, assoc_srg)
    
    # 计算显著性
    p = len(assocs_srg[assocs_srg >= assoc]) / rounds
    indep = False if p < alpha else True  # 独立: 0, 不独立: 1
    return assoc, (p, indep, assocs_srg)