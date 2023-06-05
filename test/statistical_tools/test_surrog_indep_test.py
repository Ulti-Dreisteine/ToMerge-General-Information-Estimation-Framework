# -*- coding: utf-8 -*-
"""
Created on 2023/04/26 17:19:33

@File -> test_surrog_indep_test.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 基于代用数据的独立性检验
"""

import arviz as az
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
sys.path.insert(0, BASE_DIR)

from setting import plt
from giefstat.statistical_tools.surrog_indep_test import exec_surrog_indep_test
from dataset.bivariate.data_generator import DataGenerator


# def _normalize(x):
#     return (x - np.min(x)) / (np.max(x) - np.min(x))


def gen_test_data(func, N, scale):
    data_gener = DataGenerator()
    x, y, _, _ = data_gener.gen_data(N, func, normalize=False)
    y_range = np.max(y) - np.min(y)
    noise = np.random.uniform(-scale * y_range, scale * y_range, y.shape)
    y_noise = y.copy() + noise
    return x, y_noise


if __name__ == "__main__":
    x, y = gen_test_data("sin_low_freq", 1000, 0.5)
    
    plt.scatter(x, y, color="w", edgecolor="k")
    
    rounds = 100
    alpha = 0.05
    
    # 二元变量独立性检验
    method = "MI-GIEF"
    assoc, (p, indep, assocs_srg) = exec_surrog_indep_test(
        x, y, method, xtype="c", ytype="c", rounds=rounds, alpha=alpha)
    az.plot_posterior(
        {f"{method}_Surrog": assocs_srg}, 
        kind="hist", 
        bins=20, 
        ref_val=assoc, 
        hdi_prob=1 - alpha * 2)
    
    # 三元变量独立性检验
    method = "CMI-GIEF"
    assoc, (p, indep, assocs_srg) = exec_surrog_indep_test(
        x, y, method, z=x.copy(), xtype="c", ytype="c", ztype="c", rounds=rounds, alpha=alpha)
    az.plot_posterior(
        {f"{method}_Surrog": assocs_srg}, 
        kind="hist", 
        bins=20, 
        ref_val=assoc, 
        hdi_prob=1 - alpha * 2)
