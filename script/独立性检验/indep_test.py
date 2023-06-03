# -*- coding: utf-8 -*-
"""
Created on 2023/06/02 14:01:59

@File -> main.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 二元变量关联度量与独立性检验
"""

import arviz as az
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
sys.path.insert(0, BASE_DIR)

from setting import plt
from statistical_significance.surrog_indep_test import exec_surrog_indep_test

if __name__ == "__main__":
    
    # ---- 载入数据集 -------------------------------------------------------------------------------
    
    from dataset.bivariate.data_generator import gen_dataset
    
    funcs = ["random", "line", "parabola", "categorical", "circular", "sin_high_freq"]
    N = 1000
    
    method = "MI-GIEF"
    rounds = 300
    alpha = 0.01
    
    _, axs = plt.subplots(3, 2, figsize=(10, 10))
    for i, func in enumerate(funcs):
        x, y = gen_dataset(func, N)
        
        # 进行关联度量和独立性检验
        assoc, (p, indep, assocs_srg) = exec_surrog_indep_test(
            x, y, method, xtype="c", ytype="c", rounds=rounds, alpha=alpha)
        
        # 画图
        ax = axs[i // 2, i % 2]
        az.plot_posterior(
            {f"{method}_Surrog": assocs_srg}, 
            kind="hist",
            bins=50,
            ref_val=assoc,
            hdi_prob=1 - alpha * 2,
            ax=ax)
        ax.set_title(f"dataset: {func}, independence detected: {indep}", fontsize=18)
        ax.set_xlabel("MI value")
    plt.tight_layout()
    
    