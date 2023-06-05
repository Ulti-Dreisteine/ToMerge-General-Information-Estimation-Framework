# -*- coding: utf-8 -*-
# sourcery skip: avoid-builtin-shadow
"""
Created on 2023/06/05 14:44:46

@File -> main.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 时延关联检测
"""

import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
sys.path.insert(0, BASE_DIR)

from setting import plt
from statistical_tools.time_delayed_association import detect_time_delayed_assoc

if __name__ == "__main__":
    
    # ---- 载入测试数据 -----------------------------------------------------------------------------

    from dataset.time_delayed.data_generator import gen_four_species

    samples = gen_four_species(N=6000)
    x, y = samples[:, 1], samples[:, 2]

    # ---- 时延关联检测 -----------------------------------------------------------------------------

    taus = np.arange(-20, 20, 1)
    _ = detect_time_delayed_assoc(x, y, taus, show=True, alpha=0.01, rounds=10)
    