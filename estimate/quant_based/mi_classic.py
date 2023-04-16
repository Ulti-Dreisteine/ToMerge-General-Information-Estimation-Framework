from pyitlib import discrete_random_variable as drv
import pandas as pd
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 2))
sys.path.insert(0, BASE_DIR)

from util import stdize_values


def discretize_series(x: np.ndarray, n: int = None, method="qcut"):
    """对数据序列采用等频分箱"""
    n = len(x) // 15 if n is None else n
    q = int(len(x) // n)

    # 分箱方式.
    if method == "qcut":
        x_enc = pd.qcut(x.flatten(), q, labels=False, duplicates="drop").flatten()  # 等频分箱
    if method == "cut":
        x_enc = pd.cut(x.flatten(), q, labels=False, duplicates="drop").flatten()  # 等宽分箱
        
    return x_enc


class MutualInfoClassic(object):
    """基于经典离散化的互信息和条件互信息计算"""
    
    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray=None):
        self.x_norm = stdize_values(x, "c")
        self.y_norm = stdize_values(y, "c")
        self.z_norm = stdize_values(z, "c") if z is not None else None
    
    def __call__(self, method="qcut"):
        x_enc = discretize_series(self.x_norm, method=method).astype(int)
        y_enc = discretize_series(self.y_norm, method=method).astype(int)
        if self.z_norm is None:
            return drv.information_mutual(x_enc, y_enc)
        
        z_enc = discretize_series(self.z_norm, method=method).astype(int)
        return drv.information_mutual_conditional(x_enc, y_enc, z_enc)
    

if __name__ == "__main__":
    
    def test_cc():
        x = np.random.normal(0, 1, 100000)
        y = np.random.normal(0, 1, 100000)
        print(f"mi = {MutualInfoClassic(x, y)()}")
        
    def test_ccc():
        x = np.random.normal(0, 1, 100000)
        y = np.random.normal(0, 1, 100000)
        z = np.random.normal(0, 1, 100000)
        print(f"cmi = {MutualInfoClassic(x, y, z)()}")
        
    test_cc()
    test_ccc()