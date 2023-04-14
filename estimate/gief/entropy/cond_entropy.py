# -*- coding: utf-8 -*-
"""
Created on 2022/09/18 16:19:57

@File -> cond_entropy.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 条件熵
"""

import numpy as np

from .marg_entropy import MargEntropy
from ..mutual_info.mi import MutualInfoGIEF


class CondEntropy(object):
    """条件熵"""
    
    def __init__(self, x: np.ndarray, xtype: str, z: np.ndarray, ztype: str):
        # note 此处不再对x, z进行标准化和类型检查, 因为这些步骤会在底层MargEntropy和MutualInfo中进行
        self.x = x.copy().reshape(len(x), -1)
        self.z = z.copy().reshape(len(z), -1)
        self.xtype = xtype
        self.ztype = ztype
        
    def __call__(self, k=3, metric="chebyshev"):
        return MargEntropy(self.x, self.xtype)(k=k, metric=metric) -\
            MutualInfoGIEF(self.x, self.xtype, self.z, self.ztype)(k=k, metric=metric)