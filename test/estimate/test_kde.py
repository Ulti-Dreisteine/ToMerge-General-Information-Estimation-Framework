# -*- coding: utf-8 -*-
"""
Created on 2023/04/14 16:37:53

@File -> test_kde.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 测试
"""

import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
sys.path.insert(0, BASE_DIR)

from estimate.kde.kde import MargEntropy, MutualInfoKDE

x = np.random.normal(0, 1, 1000)
y = np.random.normal(0, 1, 1000)
xy = np.c_[x, y]

# ---- 测试 -----------------------------------------------------------------------------------------

# 信息熵
self = MargEntropy(xy)
entropy = self()
print(f"entropy: {round(entropy, 4)}")

# 边际熵
self = MutualInfoKDE(x, y)
mi = self()
print(f"mi: {round(mi, 4)}")