# import matplotlib.pyplot as plt
from typing import List
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 2))
sys.path.insert(0, BASE_DIR)

if "plt" not in dir():
    import matplotlib.pyplot as plt
    
from util import stdize_values


def cal_area(bounds: List[tuple], D) -> float:
    """计算面积"""
    area = 1.0
    for i in range(D):
        bd = bounds[i]
        area *= max(bd) - min(bd)
    return area


def cal_equi_partition_thres(arr: np.ndarray) -> List[float]:
    """获取各维度上等边际概率(即等边际样本数)分箱的阈值, 维数为2
    :param arr: 待分析二维样本数组
    """
    # todo: 检查arr可否为空
    N, D = arr.shape
    
    # 如果只有1个样本, 就直接返回样本各维度的值
    if N == 1:
        return list(arr.flatten())
    
    # 如果多于1个样本, 则偶数时取中间两个值的均值, 奇数时取中间值两侧值的均值
    part_idx = N // 2  # 离散化位置idx
    part_thres = []
    for i in range(D):
        series_srt = np.sort(arr[:, i])  # 对应维度值升序排列
        if N % 2 == 0:  # 以均值划分
            marginal_part_value = (series_srt[part_idx - 1] + series_srt[part_idx]) / 2
        else:
            marginal_part_value = (series_srt[part_idx - 1] + series_srt[part_idx + 1]) / 2
        part_thres.append(marginal_part_value)
    return part_thres


class Cell(object):
    """样本数据格对象"""
    
    def __init__(self, arr: np.ndarray=None) -> None:
        # if (arr is None) or (len(arr) < N_min):
        if (arr is None) or (len(arr) == 0):
            self.is_empty = True
            self.arr = np.array([]).reshape(0, 2)
            self.N, self.D = self.arr.shape
        else:
            arr = arr.copy()
            self.is_empty = False
            self.arr = arr.reshape(len(arr), -1)
            self.N, self.D = self.arr.shape
            if self.D != 2:
                raise RuntimeError(f"现有代码仅支持两个一维数据间的计算, self.D=2, 但实际self.D={self.D}")
    
    def def_cell_bounds_cal_area(self, bounds: List[tuple]):
        """手动定义cell的边界并计算对应覆盖的面积
        :param bounds: 边界值list, 如[(x_min, x_max), (y_min, y_max)]
        """
        self.bounds = bounds
        self.area = cal_area(self.bounds, self.D)
    
    def cal_proba_dens(self, N_total: int) -> float:
        """计算以样本数计的概率密度"""
        return 0.0 if self.area == 0.0 else self.N / (N_total * self.area)

    def get_marginal_partition_thres(self):
        self.part_thres = cal_equi_partition_thres(self.arr)
        
    def exec_partition(self):
        """执行等频率离散化, 要求样本self.N > 0"""
        assert self.N > 0
        
        # 先在x方向上分为左右两部分.
        part_arr_l = self.arr[
            np.where((self.arr[:, 0] < self.part_thres[0]) &
                     (self.arr[:, 0] >= self.bounds[0][0]))
        ]
        part_arr_r = self.arr[
            np.where((self.arr[:, 0] >= self.part_thres[0])
                     & (self.arr[:, 0] <= self.bounds[0][1]))
        ]

        # 再在y方向上继续切分.
        part_arr_ul = part_arr_l[np.where(
            (part_arr_l[:, 1] >= self.part_thres[1]) & (part_arr_l[:, 1] <= self.bounds[1][1]))]
        part_arr_ll = part_arr_l[np.where(
            (part_arr_l[:, 1] < self.part_thres[1]) & (part_arr_l[:, 1] >= self.bounds[1][0]))]

        part_arr_ur = part_arr_r[np.where(
            (part_arr_r[:, 1] >= self.part_thres[1]) & (part_arr_r[:, 1] <= self.bounds[1][1]))]
        part_arr_lr = part_arr_r[np.where(
            (part_arr_r[:, 1] < self.part_thres[1]) & (part_arr_r[:, 1] >= self.bounds[1][0]))]

        cell_ul, cell_ur, cell_ll, cell_lr = Cell(part_arr_ul), Cell(part_arr_ur), \
            Cell(part_arr_ll), Cell(part_arr_lr)

        # 确定边界.
        (xl, xu), (yl, yu) = self.bounds
        x_thres, y_thres = self.part_thres
        cell_ul.def_cell_bounds_cal_area([(xl, x_thres), (y_thres, yu)])
        cell_ur.def_cell_bounds_cal_area([(x_thres, xu), (y_thres, yu)])
        cell_ll.def_cell_bounds_cal_area([(xl, x_thres), (yl, y_thres)])
        cell_lr.def_cell_bounds_cal_area([(x_thres, xu), (yl, y_thres)])
        return cell_ul, cell_ur, cell_ll, cell_lr
    
    def show(self, N_total: int, pdf_ub: float=1e4, linewidth: float = 0.5):
        (xl, xu), (yl, yu) = self.bounds
        # plt.fill_between([xl, xu], [yl, yl], [yu, yu], alpha=self.cal_proba_dens(N_total) / pdf_ub, facecolor="grey")
        plt.plot([xl, xu], [yl, yl], '-', c='k', linewidth=linewidth)
        plt.plot([xu, xu], [yl, yu], '-', c='k', linewidth=linewidth)
        plt.plot([xu, xl], [yu, yu], '-', c='k', linewidth=linewidth)
        plt.plot([xl, xl], [yu, yl], '-', c='k', linewidth=linewidth)
        

# 递归离散化.
def _try_partition(cell: Cell, p_eps: float, N_total: int):
    """尝试对Cell进行一次分裂, 若样本数少于规定阈值或概率密度收敛, 则返回为四个空cell, 否则正常分裂"""
    # 尝试分裂一下, 并检查分裂效果.
    proba_dens = cell.cal_proba_dens(N_total)

    cell.get_marginal_partition_thres()
    cell_ul, cell_ur, cell_ll, cell_lr = cell.exec_partition()

    # 所有子cell的PDF均收敛则为收敛
    is_proba_dens_converged = all(np.abs(c.cal_proba_dens(N_total) - proba_dens) / proba_dens <= p_eps \
        for c in [cell_ul, cell_ur, cell_ll, cell_lr])

    return (Cell(), Cell(), Cell(), Cell()) if is_proba_dens_converged else (cell_ul, cell_ur, cell_ll, cell_lr)


def recursively_partition(cell: Cell, N_total: int, min_samples_split: int = 30, p_eps: float = 1e-3):
    """对一个cell进行递归离散化

    :param cell: 初始cell
    :param p_eps: 子cell概率与父cell相对偏差阈值, 如果所有都小于该值则终止离散化, defaults to 1e-3
    """
    leaf_cells = []

    def _partition(cell):
        if cell.N >= min_samples_split:
            part_ul, part_ur, part_ll, part_lr = _try_partition(cell, p_eps, N_total)
            
            if all(p.is_empty for p in [part_ul, part_ur, part_ll, part_lr]):
                leaf_cells.append(cell)
            else:
                _partition(part_ul)
                _partition(part_ur)
                _partition(part_ll)
                _partition(part_lr)
        else:
            leaf_cells.append(cell)

    _partition(cell)
    return leaf_cells


def exec_partition(x: np.ndarray, y: np.ndarray):
    """对X和Y数据在二维平面进行离散化"""
    x, y = x.flatten(), y.flatten()
    assert len(x) == len(y)
    N_total = len(x)
    
    # 2.1 预处理
    arr = np.c_[x, y]
    arr_norm = stdize_values(arr, dtype="c")
    
    # 2.2 初始化Cell
    cell = Cell(arr_norm)
    cell.def_cell_bounds_cal_area([(0.0, 1.0), (0.0, 1.0)])
    
    # 2.3 执行自适应划分
    leaf_cells = recursively_partition(cell, N_total, min_samples_split=30, p_eps=1e-3)
    
    return leaf_cells, arr_norm