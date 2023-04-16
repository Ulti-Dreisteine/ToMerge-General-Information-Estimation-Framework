import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 2))
sys.path.insert(0, BASE_DIR)

from util import stdize_values
from quant_based._quant_darbellay import exec_partition, Cell


class MutualInfoDarbellay(object):
    """基于Darbellay自适应分箱的互信息计算"""
    
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x_norm = stdize_values(x, "c")
        self.y_norm = stdize_values(y, "c")
        try:
            assert self.x_norm.shape[1] == 1
            assert self.y_norm.shape[1] == 1
        except Exception as e:
            raise ValueError("不支持非一维变量间的MI计算") from e
    
    def __call__(self):
        leaf_cells, arr_norm = exec_partition(self.x_norm, self.y_norm)
        N_total, _ = arr_norm.shape
        
        # 计算互信息.
        n_leafs = len(leaf_cells)

        mi = 0.0
        for i in range(n_leafs):
            cell = leaf_cells[i]  # type: Cell
            (xl, xu), (yl, yu) = cell.bounds

            Nxy = len(cell.arr)
            Nx = len(
                np.where((arr_norm[:, 0] >= xl) & (arr_norm[:, 0] < xu))[0])
            Ny = len(
                np.where((arr_norm[:, 1] >= yl) & (arr_norm[:, 1] < yu))[0])
            gain = Nxy * np.log(Nxy / Nx / Ny)
            mi += gain
        
        mi = round(mi / N_total + np.log(N_total), 4)
        return mi
    

if __name__ == "__main__":
    
    def test_cc():
        x = np.random.normal(0, 1, 10000)
        y = np.random.normal(0, 1, 10000)
        print(f"mi = {MutualInfoDarbellay(x, y)()}")
        
    test_cc()