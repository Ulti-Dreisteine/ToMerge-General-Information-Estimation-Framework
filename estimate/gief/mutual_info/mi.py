# -*- coding: utf-8 -*-
"""
Created on 2022/09/18 15:52:28

@File -> mi.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 互信息
"""

from numpy import log
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 4))
sys.path.insert(0, BASE_DIR)

from estimate.setting import DTYPES, BASE
from estimate.util import stdize_values
from ._kraskov import kraskov_mi as cal_kraskov_mi
from ._ross import cal_ross_mi
from ..entropy.marg_entropy import MargEntropy


def _cal_mi_cc(x, y, k):
    return cal_kraskov_mi(x, y, k=k, base=BASE, alpha=0)
    
    
def _cal_mi_cd(x, y, k, metric="chebyshev", base=BASE):
    return cal_ross_mi(x, y, k, metric, base)
    

def _cal_mi_dd(x, y):
    return MargEntropy(x, "d")() + MargEntropy(y, "d")() - MargEntropy(np.c_[x, y], "d")()


class MutualInfoGIEF(object):
    """两一维或多维变量间的互信息"""
    
    def __init__(self, x: np.ndarray, xtype: str, y: np.ndarray, ytype: str):
        assert xtype in DTYPES
        assert ytype in DTYPES
        self.x_norm = stdize_values(x, xtype)
        self.y_norm = stdize_values(y, ytype)
        self.xtype = xtype
        self.ytype = ytype
        
    def __call__(self, k: int=3, metric: str="chebyshev"):
        if (self.xtype == "d") & (self.ytype == "d"):
            return _cal_mi_dd(self.x_norm, self.y_norm) / log(BASE)
        elif (self.xtype == "c") & (self.ytype == "d"):
            return _cal_mi_cd(self.x_norm, self.y_norm, k, metric=metric) / log(BASE)
        elif (self.xtype == "d") & (self.ytype == "c"):
            return _cal_mi_cd(self.y_norm, self.x_norm, k, metric=metric) / log(BASE)
        elif (self.xtype == "c") & (self.ytype == "c"):
            return _cal_mi_cc(self.x_norm, self.y_norm, k) / log(BASE)
        else:
            raise RuntimeError("")


if __name__ == "__main__":
    def test_cc():
        x = np.random.normal(0, 1, 10000)
        y = np.random.normal(0, 1, 10000)
        
        self = MargEntropy(x, "c")
        Hx = self()
        
        self = MargEntropy(y, "c")
        Hy = self()
        
        self = MargEntropy(np.c_[x, y], "c")
        Hxy = self()
        
        self = MutualInfoGIEF(x, "c", y, "c")
        MI = self()
        
        print(f"Hx + Hy - Hxy: {Hx + Hy - Hxy} \tMI: {MI}")
        
    def test_dd():
        x = np.random.randint(0, 5, 10000)
        y = np.random.randint(0, 5, 10000)
        
        self = MargEntropy(x, "d")
        Hx = self()
        
        self = MargEntropy(y, "d")
        Hy = self()
        
        self = MargEntropy(np.c_[x, y], "d")
        Hxy = self()
        
        self = MutualInfoGIEF(x, "d", y, "d")
        MI = self()
        
        print(f"Hx + Hy - Hxy: {Hx + Hy - Hxy} \tMI: {MI}")
        
    def test_cd():
        x = np.random.normal(0, 1, 10000)
        y = np.random.randint(0, 5, 10000)
        
        self = MutualInfoGIEF(x, "c", y, "d")
        MI = self()
        
        print(f"MI: {MI}")
        
    def test_dc():
        x = np.random.randint(0, 5, 10000)
        y = np.random.normal(0, 1, 10000)
        
        self = MutualInfoGIEF(x, "d", y, "c")
        MI = self()
        
        print(f"MI: {MI}")
        
    print("test cc")
    test_cc()
    
    print("test dd")
    test_dd()
    
    print("test cd")
    test_cd()
    
    print("test dc")
    test_dc()