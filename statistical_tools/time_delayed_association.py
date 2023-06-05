# -*- coding: utf-8 -*-
"""
Created on 2023/06/05 15:44:42

@File -> td_assoc_detection.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 时延关联检测
"""

import matplotlib.pyplot as plt
import numpy as np
import logging

from .util import build_td_series
from .surrog_indep_test import exec_surrog_indep_test


def detect_time_delayed_assoc(x, y, taus, xtype="c", ytype="c", max_size_bt=1000, method="MI-GIEF", 
                              show=False, **kwargs):
    """双变量间的时延关联检测

    :param x: x样本, np.ndarray
    :param y: y样本, np.ndarray
    :param taus: 待计算时延数组, np.ndarray
    :param xtype: x的数值类型, str, defaults to "c"
    :param ytype: y的数值类型, str, defaults to "c"
    :param max_size_bt: 重采样计算中的最大样本数, defaults to 1000
    :param method: 所使用的关联度量方法, defaults to "MI-GIEF"
    :param show: 是否显示结果, bool, defaults to False
    :param **kwargs: 如alpha, rounds等, 见exec_surrog_indep_test中的参数设置
    """
    x, y = x.flatten(), y.flatten()
    N = len(x)

    # 检查样本长度
    try:
        assert N == len(y)
    except Exception as e:
        raise RuntimeError("x的长度与y不相等") from e

    # 抽样样本量参数设置, 后续计算需以此为基础统一计算
    N_drop_tau_max = N - np.max(np.abs(taus))  # 时延计算中的最低样本量
    if N_drop_tau_max <= max_size_bt:
        size_bt = N_drop_tau_max
    else:
        logging.warning(f"采用默认max_size_bt={max_size_bt}作为size_bt, 用于限制代用数据重采样规模")
        size_bt = max_size_bt
    
    # 逐时延计算
    td_assocs = np.array([])
    td_indeps = np.array([])
    for tau in taus:
        # 构造时延序列
        x_td, y_td = build_td_series(x, y, tau)
        
        # 进行关联计算
        assoc, (_, indep, _) = exec_surrog_indep_test(
            x_td, y_td, method, xtype=xtype, ytype=ytype, size_bt=size_bt, **kwargs)
        
        td_assocs = np.append(td_assocs, assoc)
        td_indeps = np.append(td_indeps, indep)
        
    # 显示结果, 红色点表示显著
    if show:
        _show_results(taus, td_assocs, td_indeps)
        
    return td_assocs, td_indeps
        

def _show_results(taus, td_assocs, td_indeps):
    plt.figure()
    plt.scatter(
        taus, td_assocs, edgecolors=["k" if p==1 else "r" for p in td_indeps], c="w", lw=2, zorder=1)
    plt.plot(taus, td_assocs, c="k", lw=1.5, zorder=0)
    plt.grid(alpha=0.3, zorder=-1)
    plt.xlabel("$\\tau$")
    plt.ylabel("MI")