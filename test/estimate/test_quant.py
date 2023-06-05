import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
sys.path.insert(0, BASE_DIR)

from giefstat.estimate.quant_based import MutualInfoClassic, MutualInfoDarbellay


# ---- 测试经典估计方法 ------------------------------------------------------------------------------

def test_mi_class_cc():
    x = np.random.normal(0, 1, 100000)
    y = np.random.normal(0, 1, 100000)
    print(f"mi = {MutualInfoClassic(x, y)()}")
    
    
def test_mi_class_ccc():
    x = np.random.normal(0, 1, 100000)
    y = np.random.normal(0, 1, 100000)
    z = np.random.normal(0, 1, 100000)
    print(f"cmi = {MutualInfoClassic(x, y, z)()}")
    
    
test_mi_class_cc()
test_mi_class_ccc()


# ---- 测试Darbellay估计 ----------------------------------------------------------------------------

def test_mi_darbellay_cc():
    x = np.random.normal(0, 1, 10000)
    y = np.random.normal(0, 1, 10000)
    print(f"mi = {MutualInfoDarbellay(x, y)()}")
    
    
test_mi_darbellay_cc()