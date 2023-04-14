# -*- coding: utf-8 -*-
# sourcery skip: avoid-builtin-shadow
"""
Created on 2022/09/02 19:58:22

@File -> data_generator.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: Four-Spiecies时延数据生成
"""

__doc__ = """
    参考文献:
    1. H. Ye, et al.: Distinguishing time-delayed causal interactions using convergent cross mapping, 2015.
"""

import numpy as np


def gen_data(rounds: int):
    X = np.ones(4).reshape(1, -1) * 0.4
    for i in range(rounds):
        x = X[i, :].copy()
        x_new = [
            x[0] * (3.9 - 3.9 * x[0]),
            x[1] * (3.6 - 0.4 * x[0] - 3.6 * x[1]),
            x[2] * (3.6 - 0.4 * x[1] - 3.6 * x[2]),
            x[3] * (3.8 - 0.35 * x[2] - 3.8 * x[3]) 
        ]
        X = np.r_[X, np.array(x_new).reshape(1, -1)]
    return X


if __name__ == "__main__":
    import sys
    import os
    
    BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
    sys.path.insert(0, BASE_DIR)
    
    from setting import plt
    
    rounds = 5000

    X = gen_data(rounds)
    
    D = X.shape[1]
    plt.figure()
    for i in range(D):
        plt.subplot(D, 1, i + 1)
        plt.plot(X[1000:1100, i])
        plt.ylabel(f"$x_{i + 1}$", rotation=0, fontsize=16)
        if i == D - 1:
            plt.xlabel("sample no.", fontsize=16)
    plt.tight_layout()
    