import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
sys.path.insert(0, BASE_DIR)

from setting import plt
from giefstat.estimate import cal_assoc
from dataset.bivariate.data_generator import gen_dataset

EPS = 1e-6

if __name__ == "__main__":
    x, y = gen_dataset("categorical", N=5000)
    
    plt.scatter(x, y)
    
    # 按照原始数据类型来处理
    mi_dc = cal_assoc(x, y, "MI-GIEF", "d", "c")
    print(f"mi_dc={mi_dc}")
    
    mi_cc = cal_assoc(x, y, "MI-GIEF", "c", "c")
    print(f"mi_cc={mi_cc}")