# -*- coding: utf-8 -*-
"""
Created on 2022/09/18 16:19:57

@File -> cond_entropy.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 条件熵
"""

import numpy as np
# import sys
# import os

# BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
# sys.path.insert(0, BASE_DIR)

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


if __name__ == "__main__":
    def test_cc():
        x = np.random.normal(0, 1, 1000)
        z = np.random.normal(0, 1, 1000)
        
        self = CondEntropy(x, "c", z, "c")
        ce = self()
        
        print(f"ce: {ce}")
        
    def test_dd():
        x = np.random.randint(0, 5, 1000)
        z = np.random.randint(0, 5, 1000)
        
        self = CondEntropy(x, "d", z, "d")
        ce = self()
        
        print(f"ce: {ce}")
        
        
    def test_cd():
        x = np.random.normal(0, 1, 1000)
        z = np.random.randint(0, 5, 1000)
        
        self = CondEntropy(x, "c", z, "d")
        ce = self()
        
        print(f"ce: {ce}")
        
    def test_dc():
        x = np.random.randint(0, 5, 1000)
        z = np.random.normal(0, 1, 1000)
        
        self = CondEntropy(x, "d", z, "c")
        ce = self()
        
        print(f"ce: {ce}")
        
    print("test cc")
    test_cc()
    
    print("test dd")
    test_dd()
    
    print("test cd")
    test_cd()
    
    print("test dc")
    test_dc()