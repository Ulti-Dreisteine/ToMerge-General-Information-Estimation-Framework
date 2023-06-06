# -*- coding: utf-8 -*-
# sourcery skip: avoid-builtin-shadow
"""
Created on 2023/04/26 16:45:20

@File -> surrog_indep_test.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 基于代用数据的独立性测试
"""

__doc__ = """
    计算关联系数并通过代用数据重采样获得结果的显著性信息
"""

import logging
import random
import numpy as np

from . import cal_general_assoc


def exec_surrog_indep_test(x, y, method, z=None, xtype=None, ytype=None, ztype=None, rounds=100, 
                           alpha=0.05, max_size_bt=1000, size_bt=None, **kwargs):
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
    :param max_size_bt: 用于自助重采样的最大样本数, defaults to 1000
    :param size_bt: 自行设定的重采样规模, int, defaults to None
    :param kwargs: cal_general_assoc方法中的关键字参数
    :return
        assoc: 关联值, float
        p: P值, float
        indep: 是否独立, bool
        assocs_srg: 代理数据关联值数组, np.ndarray
    """
    # 如果样本量小, 则全部代入计算; 否则, 选择max_size_bt的样本进行计算
    if size_bt is None:
        if len(x) <= max_size_bt:
            size_bt = len(x)
        else:
            logging.warning(
                f"exec_surrog_indep_test: 采用默认max_size_bt={max_size_bt}作为size_bt, \
                用于限制代用数据重采样规模")
            size_bt = max_size_bt
        
    # 计算关联系数
    # TODO: 通过随机抽样获得关联值分布, 有放回or无放回?
    if z is None:
        assoc = cal_general_assoc(x[:size_bt], y[:size_bt], z, method, xtype, ytype, ztype, **kwargs)
    else:
        assoc = cal_general_assoc(x[:size_bt], y[:size_bt], z[:size_bt], method, xtype, ytype, ztype, **kwargs)

    # 计算背景值
    idxs = np.arange(len(x))
    assocs_srg = np.array([])
    for _ in range(rounds):
        # 基于代用数据获得背景值
        idxs_bt = random.choices(idxs, k=size_bt)  # NOTE: 有放回抽样
        x_bt, y_bt = x[idxs_bt], y[idxs_bt]
        x_srg = np.random.permutation(x_bt)  # 随机重排获得代用数据
        if z is None:
            assoc_srg = cal_general_assoc(x_srg, y_bt, None, method, xtype, ytype, ztype, **kwargs)
        else:
            z_bt = z[idxs_bt]
            assoc_srg = cal_general_assoc(x_srg, y_bt, z_bt, method, xtype, ytype, ztype, **kwargs)

        assocs_srg = np.append(assocs_srg, assoc_srg)

    # 计算显著性
    p = len(assocs_srg[assocs_srg >= assoc]) / rounds
    indep = p >= alpha
    return assoc, (p, indep, assocs_srg)