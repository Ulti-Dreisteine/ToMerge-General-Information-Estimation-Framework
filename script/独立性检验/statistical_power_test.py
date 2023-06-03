# -*- coding: utf-8 -*-
"""
Created on 2023/06/02 14:33:27

@File -> statistical_power_test.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 统计效能测试
"""

# from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from collections import defaultdict
import pandas as pd
import numpy as np
import itertools
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
sys.path.insert(0, BASE_DIR)

from setting import plt
from dataset.bivariate.data_generator import DataGenerator
from statistical_significance.surrog_indep_test import exec_surrog_indep_test


def gen_test_data(func, N, scale):
    """根据样本量和噪声级别生成测试数据"""
    data_gener = DataGenerator()
    x, y, _, _ = data_gener.gen_data(N, func, normalize=False)
    y_range = np.max(y) - np.min(y)
    noise = np.random.uniform(-scale * y_range, scale * y_range, y.shape)
    y += noise
    return x, y


def cal_assoc_rate(func, N, scale, method="MI-GIEF", alpha=0.01, rounds=100, **kwargs):
    """计算数据关系为func时在特定N和scale对应的关联检出率

    :param func: DataGenerator中采用的数据关系名, str
    :param N: 样本量, int
    :param scale: 在y值上加入的噪音系数, float
    :param method: 采用的关联度量方法, str, defaults to "MI-GIEF"
    :param alpha: 显著性水平, float, defaults to 0.01
    :param rounds: 重复采样-测试轮数, int, defaults to 100
    :**kwargs: exec_surrog_indep_test中的关键字参数
    """
    assoc_times = 0
    for _ in range(rounds):
        x, y = gen_test_data(func, N, scale)
        
        # 进行关联度量和独立性检验
        _, (_, indep, _) = exec_surrog_indep_test(
            x, y, method, xtype="c", ytype="c", rounds=rounds, alpha=alpha, **kwargs)

        if not indep:
            assoc_times += 1
            
    return assoc_times / rounds


if __name__ == "__main__":
    method = "MI-GIEF"
    alpha = 0.01
    rounds = 100

    Ns = [50, 100, 200, 500, 1000]
    scales = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0]
    
    funcs = ["random", "line", "parabola", "circular", "sin_high_freq", "exp_base_2"]

    # ---- 各数据关系统计效力测试 --------------------------------------------------------------------
    
    recal_rates = False
    if recal_rates:
        for i, func in enumerate(funcs):
            print(f"\ndataset: {func}")
            results = defaultdict(dict)
            for N, scale in itertools.product(Ns, scales):
                print(f"computing {N}, {scale} ...")
                _rate = cal_assoc_rate(func, N, scale)
                results[N][scale] = _rate  
            results_df = pd.DataFrame.from_dict(results)
            
            # 保存结果至本地
            np.save(f"file/{func}_rate_results.npy", results_df.values)
        
    # 绘制曲面图
    Ns_grid, scales_grid = np.meshgrid(Ns, scales)
    
    fig = plt.figure(figsize=(16, 8))
    for i, func in enumerate(funcs):
        values = np.load(f"file/{func}_rate_results.npy")
        
        ax = fig.add_subplot(2, 3, i + 1, projection="3d")
        ax.plot_surface(
            scales_grid, 
            Ns_grid, 
            values,
            cmap=cm.viridis,
            linewidth=0,
            antialiased=True)
        # ax.set_title(f"dataset: {func}", fontsize=16)
        ax.set_xlabel("noise scale $\sigma$", fontsize=14)
        ax.set_ylabel("sample size $N$", fontsize=14)
        ax.set_zlabel("assoc rate detected", fontsize=14)
        ax.text(0, 0, np.max(values) * 1.5 - 0.5 * np.min(values), func, fontsize=16)
        # ax.set_zlim([-0.1, 1.1])
    plt.tight_layout()
    plt.savefig("img/统计效能测试结果.png", dpi=450)
    
    # ---- 绘制各数据关系分布 ------------------------------------------------------------------------
    
    for func in funcs:
        x, y = gen_test_data(func, N=2000, scale=0.1)
        # if func == "exp_base_2":
        #     fs=(1.4, 1.2)
        # else:
        #     fs=(1.2, 1.2)
        plt.figure(figsize=(1.2, 1.2))
        plt.scatter(x, y, s=3, alpha=1, c="k", lw=0.1, facecolor="w")
        plt.xticks([], [])
        plt.yticks([], [])
        # plt.xlabel("variable $x$", fontsize=14)
        # plt.ylabel("variable $y$", fontsize=14)
        # plt.ticklabel_format(style="sci", scilimits=(-2, 3), axis="y")
        # plt.tight_layout()
        # plt.show()
        plt.savefig(f"img/data_{func}.png", dpi=450)