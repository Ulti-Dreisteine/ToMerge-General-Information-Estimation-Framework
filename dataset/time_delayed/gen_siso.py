# -*- coding: utf-8 -*-
"""
Created on 2021/03/21 14:31

@Project -> File: rmic-chemical-process-causality-analysis -> gen_siso.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 初始化
"""

import pandas as pd
import numpy as np

# ---- SISO 过程函数 --------------------------------------------------------------------------------

STD = 0.01  # 噪声的标准差


def gen_linear_static_outputs(x, tau_0: int = 3):
    np.random.seed(0)
    y = np.hstack((
        np.zeros(tau_0),
        x[: -tau_0] + np.random.normal(0, STD, len(x) - tau_0)
    ))
    return x[tau_0:], y[tau_0:]


def gen_linear_dynamic_outputs(x, tau_0: int = 3, tau_1: int = 1):
    y_0 = np.hstack((np.zeros(tau_0), x[: -tau_0]))
    y_1 = np.hstack((np.zeros(tau_1), y_0[: -tau_1]))
    y_0, y_1 = y_0[tau_0 + tau_1:], y_1[tau_0 + tau_1:]
    y = 0.2 * y_1 / (y_0 - 0.5 * y_1)
    np.random.seed(0)
    y += np.random.normal(0, STD, len(x) - tau_0 - tau_1)
    
    return x[tau_0 + tau_1:], y


def gen_nonlinear_static_outputs(x, tau_0: int = 3, p: int = 2):
    np.random.seed(0)
    y = np.hstack((
        np.zeros(tau_0),
        np.power(x[: -tau_0], p) + np.random.normal(0, STD, len(x) - tau_0)
    ))
    return x[tau_0:], y[tau_0:]


def gen_nonlinear_dynamic_outputs(x, tau_0: int = 3, tau_1: int = 1, p: int = 2):
    y_0 = np.hstack((np.zeros(tau_0), np.power(x[: -tau_0], p)))
    y_1 = np.hstack((np.zeros(tau_1), y_0[: -tau_1]))
    y_0, y_1 = y_0[tau_0 + tau_1:], y_1[tau_0 + tau_1:]
    y = 0.2 * y_1 / (y_0 - 0.5 * y_1)
    np.random.seed(0)
    y += np.random.normal(0, STD, len(x) - tau_0 - tau_1)
    return x[tau_0 + tau_1:], y


# ---- 生成数据集 -----------------------------------------------------------------------------------

samples_n = 2100
N = 2000
np.random.seed(1)

x_0 = np.random.normal(0.0, 9.0, size = samples_n)

SISO_data = pd.DataFrame(range(N), columns = ["No."])
for i in range(4):
    x, y, label = None, None, None
    if i == 0:
        label = "linear_static"
        x, y = gen_linear_static_outputs(x_0)
    elif i == 1:
        label = "linear_dynamic"
        x, y = gen_linear_dynamic_outputs(x_0)
    elif i == 2:
        label = "nonlinear_static"
        x, y = gen_nonlinear_static_outputs(x_0, p = 2)
    elif i == 3:
        label = "nonlinear_dynamic"
        x, y = gen_nonlinear_dynamic_outputs(x_0, p = 2)
    d = pd.DataFrame(np.vstack((x, y)).T, columns=[f"x_{label}", f"y_{label}"])
    SISO_data = pd.concat([SISO_data, d.head(N)], axis = 1)
SISO_data.to_csv("siso.csv", index = False)


if __name__ == "__main__":
    import sys
    import os

    BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
    sys.path.insert(0, BASE_DIR)

    from setting import plt

    plt.plot(SISO_data.values[:,])
