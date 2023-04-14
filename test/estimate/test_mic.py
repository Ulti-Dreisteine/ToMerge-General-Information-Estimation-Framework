import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
sys.path.insert(0, BASE_DIR)

from estimate.mic import MIC, CMIC

def test_cc():
    x = np.random.normal(0, 1, 1000)
    y = np.random.normal(0, 1, 1000)
    xy = np.c_[x, y]
    print(f"mic = {MIC(xy, x)()}")
    
def test_ccc():
    x = np.random.normal(0, 1, 1000)
    y = np.random.normal(0, 1, 1000)
    z = np.random.normal(0, 1, 1000)
    xy = np.c_[x, y]
    # xz = np.c_[x, z]
    print(f"cmic = {CMIC(xy, x, y)()}")

test_cc()
test_ccc()