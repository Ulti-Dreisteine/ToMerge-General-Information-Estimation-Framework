# -*- coding: utf-8 -*-
# sourcery skip: avoid-builtin-shadow
"""
Created on 2022/08/17 18:22:01

@File -> data_generator.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 三变量数据生成
"""

import numpy as np


def normalize(x: np.ndarray):
    x_min, x_max = np.min(x.copy()), np.max(x.copy())
    if x_max == x_min:
        raise RuntimeError("x_max == x_min")
    return (x - x_min) / (x_max - x_min)


# 基本函数关系
# NOTE: 这里使用了eval()函数实现对应每个函数的采样

def _M1(x1, x2, e1, e2, z):
    return x1 + z + e1, x2 + z + e2


def _M2(x1, _, e1, e2, z):
    return x1 + z + e1, np.power(x1, 2) + z + e2


def _M3(x1, _, e1, e2, z):
    return x1 + z + e1, 0.5 * np.sin(x1 * np.pi) + z + e2


def _M4(x1, x2, e1, e2, z):
    return x1 + z + e1, x1 + x2 + z + e2


def _M5(x1, x2, e1, e2, z):
    return np.sqrt(np.abs(x1 * z)) + z + e1, 0.25 * np.power(x1, 2) * np.power(x2, 2) + x2 + z + e2


def _M6(x1, x2, e1, e2, z):
    return np.log(np.abs(x1 * z) + 1) + z + e1, 0.5 * np.power(x1, 2) * z + x2 + z + e2


class DataGenerator(object):
    """
    数据生成器
    
    ref: "A Distribution Free Conditional Independence Test with Applications to Causal Discovery"
    """
    
    def __init__(self):
        pass
    
    def gen_data(self, N: int, func: str, norm: bool=True):
        """数据生成

        :param N: 样本量
        :param func: 数据方程
        :param normalize: 是否归一化
        """
        
        # 生成底层样本
        x1 = np.random.normal(0, 1, N)
        x2 = np.random.normal(0, 1, N)
        e1 = np.random.random(N) * 1e-06
        e2 = np.random.random(N) * 1e-06
        z = np.random.normal(0, 1, N)
        
        # 生成数据
        f = eval(f"_{func}")
        x, y = f(x1, x2, e1, e2, z)
        
        if norm:
            x, y, z = normalize(x), normalize(y), normalize(z)
        
        return x, y, z
   
   
if __name__ == "__main__":
    import sys
    import os

    BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
    sys.path.insert(0, BASE_DIR)
    
    from setting import plt
    
    # func = "M2"
    
    # ---- 画各数据集中X和Y关于Z的条件分布图 ----------------------------------------------------------
    
    N = 2000
    
    plt.figure(figsize=(9, 6.5))
    for i in range(6):
        func = f"M{i + 1}"
        x, y, z = DataGenerator().gen_data(N, func)
        idxs_a, idxs_b = np.argwhere(z < 0.5).flatten(), np.argwhere(z >= 0.5).flatten()
        plt.subplot(2, 3, i + 1)
        plt.scatter(x[idxs_a], y[idxs_a], c="r", marker="+", s=50, lw=1., alpha=0.2)
        plt.scatter(x[idxs_b], y[idxs_b], c="b", marker="+", s=50, lw=1., alpha=0.2)
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.title(func)
        plt.legend(["$z<0.5$", "$z\\geq0.5$"], loc="lower right")
    plt.tight_layout()