# -*- coding: utf-8 -*-
"""
Created on 2021/03/18 11:16

@Project -> File: refined-association-process-causality-analysis -> data_generator.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 双数据生成
"""

from sklearn.preprocessing import MinMaxScaler
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
sys.path.insert(0, BASE_DIR)

# NOTE: 函数名太多就用*号概括了. 这里使用了eval()函数实现对应每个函数的采样
from dataset.bivariate import *
from mod.data_process.numpy import random_sampling

FUNC_NAMES = [
    # 连续值.
    "linear_periodic_low_freq",
    "linear_periodic_med_freq",
    "linear_periodic_high_freq",
    "linear_periodic_high_freq_2",
    "non_fourier_freq_cos",
    "cos_high_freq",
    "cubic",
    "cubic_y_stretched",
    "l_shaped",  # association != 1.0
    "exp_base_2",
    "exp_base_10",
    "line",
    "parabola",
    "random",  # association != 1.0
    "non_fourier_freq_sin",
    "sin_low_freq",
    "sin_high_freq",
    "sigmoid",
    "vary_freq_cos",
    "vary_freq_sin",
    "spike",
    "lopsided_l_shaped",  # association hard = 1
    
    # 离散值.
    "categorical",  # 非连续值
    
    # 圆圈
    "circular"
]


def min_max_norm(x: np.ndarray):
    return MinMaxScaler().fit_transform(x)


class DataGenerator(object):
    """
    数据生成器, 输出负梯度采样后的结果

    Reference:
    1. D.N. Reshef, Y.A. Reshef, et al.: "Supporting Online Material for Detecting Novel Associations
            in Large Data Sets" (Table S3), Science, 2012.
    """
    
    def __init__(self, N_ticks: int=int(1e4)):
        self.N_ticks = N_ticks  # ! 该值太大会影响计算效率
        self.func_groups = {
            0: [
                "linear_periodic_low_freq", "linear_periodic_med_freq", "linear_periodic_high_freq",
                "linear_periodic_high_freq_2", "non_fourier_freq_cos", "cos_high_freq", "l_shaped",
                "line", "random", "non_fourier_freq_sin", "sin_low_freq", "sin_high_freq", "sigmoid",
                "vary_freq_cos", "vary_freq_sin", "spike", "lopsided_l_shaped"
            ],
            1: ["cubic", "cubic_y_stretched"],
            2: ["exp_base_2", "exp_base_10"],
            3: ["parabola"],
            4: ["categorical"],
            5: ["circular"]
        }
        
    
    def _init_x_ticks(self, func: str) -> np.ndarray:
        if func in self.func_groups[0]:
            h = 1.0 / self.N_ticks
            return np.arange(0.0 - h, 1.0 + 2 * h, h)  # 这里多一个h是为了抵消后面差分h变少
        elif func in self.func_groups[1]:
            h = 2.4 / self.N_ticks
            return np.arange(-1.3 - h, 1.1 + 2 * h, h)
        elif func in self.func_groups[2]:
            h = 10.0 / self.N_ticks
            return np.arange(0.0 - h, 10.0 + 2 * h, h)
        elif func in self.func_groups[3]:
            h = 1.0 / self.N_ticks
            return np.arange(-0.5 - h, 0.5 + 2 * h, h)
        elif func in self.func_groups[4]:
            # return np.random.randint(1, 6, self.N_ticks, dtype = int)  # 随机生成1~5的随机整数
            return np.random.randint(1, 5, self.N_ticks, dtype = int)  # 随机生成1~5的随机整数
        else:
            raise RuntimeError(f"Invalid func = \'{func}\'")
    
    def gen_data(self, N: int, func: str, normalize: bool = False):
        """这里对数据进行了采样"""
        if N > self.N_ticks:
            raise ValueError("self.N_ticks < N, 减少N或增加self.N_ticks")
        
        if func in self.func_groups[5]:
            theta = np.linspace(0, 2 * np.pi, N)
            x, y = circular(theta)
        else:
            x_ticks = self._init_x_ticks(func)
            y_ticks = eval(f"{func}")(x_ticks)
            
            if func in self.func_groups[4]:
                arr = random_sampling(np.vstack((x_ticks, y_ticks)).T, N)
                x, y = arr[:, 0], arr[:, 1]
            else:
                y_derivs_l = y_ticks[1:-1] - y_ticks[:-2]
                y_derivs_r = y_ticks[2:] - y_ticks[1:-1]
                p_derivs = np.abs((y_derivs_l + y_derivs_r) / 2)
                p_derivs = p_derivs / np.sum(p_derivs)
                x_ticks_s = x_ticks.copy()[1:-1]
                y_ticks_s = y_ticks.copy()[1:-1]
                x = np.array([x_ticks_s[0], x_ticks_s[-1]])
                y = np.array([y_ticks_s[0], y_ticks_s[-1]])
                x_ = np.random.choice(x_ticks_s, size=N - 2, replace=True, p=list(p_derivs))
                y_ = eval(f"{func}")(x_)
                x = np.hstack((x, x_))
                y = np.hstack((y, y_))
        
        # 是否归一化
        if normalize:
            x_norm = min_max_norm(x)
            y_norm = min_max_norm(y)
        else:
            x_norm, y_norm = None, None
            
        return x, y, x_norm, y_norm
    
    
def gen_dataset(func: str, N: int):
    """生成未归一化的数据集"""
    generator = DataGenerator()
    x, y, _, _ = generator.gen_data(N, func)
    return x, y
        

if __name__ == "__main__":
    from setting import plt
    
    N = 500
    func = "circular"
    
    x, y = gen_dataset(func, N)
    # self = DataGenerator()
    # x, y, _, _ = self.gen_data(N, func)
    
    plt.figure()
    plt.scatter(x, y, s = 12)
    
    
