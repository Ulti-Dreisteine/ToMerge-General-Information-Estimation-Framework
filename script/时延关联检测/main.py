# -*- coding: utf-8 -*-
# sourcery skip: avoid-builtin-shadow
"""
Created on 2023/06/04 15:08:10

@File -> main.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 时延关联检测
"""

__doc__ = """
    逐taus内每个时延计算:
        在每个时延tau:
        1. 构造时延样本x_td和y_td
        2. 在x_td基础上构造代用样本x_bg
        3. 使用cal_bootstrap_coeff分别计算x_td和x_bg与y_td之间的关联系数和分布
    汇总所有时延上的结果, 并画图
"""

from collections import defaultdict
import numpy as np
import warnings
import random
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
sys.path.insert(0, BASE_DIR)

from setting import plt
from dataset.time_delayed.data_generator import gen_four_species
from estimate import cal_assoc
from statistical_significance.bootstrap_coeff import cal_bootstrap_coeff

from script.时延关联检测.util import gen_time_delayed_series, gen_surrog_data


def cal_assoc_at_tau(x, y, tau, method, xtype, ytype, bt_rounds=10, surrog=True, max_size_bt=3000):
    """计算特定tau位置的关联值

    :param x: x样本, np.ndarray
    :param y: y样本, np.ndarray
    :param tau: 时延值, int
    :param method: 所使用的关联度量方法, str
    :param xtype: x的变量类型, str
    :param ytype: y的变量类型, str
    :param bt_rounds: 用于获得显著性的自助法测试轮数, int, defaults to 100
    :param surrog: 是否通过代用数据获得背景值信息, bool
    :param max_size_bt: 用于自助重采样的最大样本数, defaults to 1000
    """
    # 生成时延序列样本
    # FIXME: 不同时延tau所得样本量不统一，干扰结果稳定性
    x_td, y_td = gen_time_delayed_series(x, y, tau)
    
    N = len(x_td)
    idxs = np.arange(N)

    # 确定自助法的最大样本数
    if len(x_td) <= max_size_bt:
        size_bt = N
    else:
        warnings.warn("采用默认max_size_bt作为size_bt")
        size_bt = max_size_bt

    results = {"assoc": defaultdict(list), "assoc_bg": defaultdict(list)}
    
    # 计算原样本和代用数据样本的关联值均值和分布
    # FIXME: 代用数据和背景值计算结果不符合：计算原关联值用boostrap，而背景值时不需要boostrap
    for _ in range(bt_rounds):
        idxs_bt = random.choices(idxs, k=size_bt)  # NOTE: 有放回抽样
        x_bt, y_bt = x_td.copy()[idxs_bt], y_td.copy()[idxs_bt]
        
        _assoc = cal_assoc(x_bt, y_bt, method, xtype, ytype)
        
        if surrog:
            x_srg = np.random.permutation(x_bt.copy())
            _assoc_bg = cal_assoc(x_srg, y_bt, method, xtype, ytype)
            
        results["assoc"]["bootstrap"].append(_assoc)
        results["assoc_bg"]["bootstrap"].append(_assoc_bg)
    
    for tag in ["assoc", "assoc_bg"]:
        results[tag]["mean"] = np.mean(results[tag]["bootstrap"])
        
    return results


if __name__ == "__main__":
    N = 6000
    samples = gen_four_species(N)
    
    # ---- 进行时延计算 ----------------------------------------------------------------------------
    
    method = "MI-GIEF"
    taus = np.arange(-2000, 2000, 100)
    
    x, y = samples[:, 0], samples[:, 1]
    
    results = []
    for tau in taus:
        print(tau)
        
        # 生成时延序列样本
        x_td, y_td = gen_time_delayed_series(x, y, tau)
        
        # break
        _r = cal_assoc_at_tau(x, y, tau, method, "c", "c")
        _r = _r["assoc"]["mean"]
        results.append(_r)
        
        break
    