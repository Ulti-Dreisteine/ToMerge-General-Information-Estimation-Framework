from pyitlib import discrete_random_variable as drv
import pandas as pd
import numpy as np
import random
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 2))
sys.path.insert(0, BASE_DIR)

from setting import plt, PROJ_CMAP  # TODO: 需自行配置

# ---- 关联系数SU -----------------------------------------------------------------------------------

# 数据离散化
def discretize_series(x: np.ndarray, q=15, method="qcut") -> np.ndarray:
    """对数据序列采用等频分箱"""
    # 分箱方式
    if method == "qcut":
        return pd.qcut(x.flatten(), q, labels=False, duplicates="drop").flatten()  # 等频分箱
    elif method == "cut":
        return pd.cut(x.flatten(), q, labels=False, duplicates="drop").flatten()  # 等宽分箱
    else:
        raise ValueError(f"unknown method {method}")


def cal_assoc(x: np.ndarray, y: np.ndarray, q=15):
    """计算SU值"""
    # 原文献使用SU, 但本文使用互信息, 且将所有变量均视为连续类型
    x_enc = discretize_series(x, q=q, method="qcut").astype(int)
    y_enc = discretize_series(y, q=q, method="qcut").astype(int)
    x_enc, y_enc = x_enc.flatten(), y_enc.flatten()
    mi = drv.information_mutual(x_enc, y_enc)
    Hx = drv.entropy(x_enc)
    Hy = drv.entropy(y_enc)
    return 2 * mi / (Hx + Hy)

# ---- 自回归 ---------------------------------------------------------------------------------------

def _gen_td_series(x: np.ndarray, tau: int):
    x, y = x.flatten(), x.flatten()
    lag_remain = np.abs(tau) % len(x)  # 求余数
    
    if tau == 0:  # 没有时滞, 那么x_td和y_td_1同时发生
        x_td, y_td = x.copy(), y.copy()
    elif tau > 0:  # 正时滞, y更早
        x_td = x.copy()[:-lag_remain]
        y_td = y.copy()[lag_remain:]
    else:  # 负时滞, x更早
        x_td = x.copy()[lag_remain:]
        y_td = y.copy()[: -lag_remain]
    return x_td, y_td


# 计算关联值分布
def _cal_assoc_dist(x, y, sub_sample_size, rounds, **kwargs):
    """计算条件关联系数分布, 每次从总样本中随机选择sub_sample_size个子样本计算, 重复rounds轮
    :param sub_sample_size: 每轮从总体中的子采样数, 较小的该值可确保样本i.i.d.
    :param rounds: 重复轮数
    """
    N = len(x)
    assocs = np.array([])
    for _ in range(rounds):
        # _idxs = np.random.permutation(np.arange(N))[:sub_sample_size]
        _idxs = random.sample(list(range(N)), sub_sample_size)
        _x, _y = x[_idxs].flatten(), y[_idxs].flatten()
        assocs = np.append(assocs, cal_assoc(_x, _y, **kwargs))
    return assocs


class SelfAssoc(object):
    """时间序列的自关联"""
    
    def __init__(self, x: np.ndarray, sub_sample_size: int, rounds: int):
        self.x = x.flatten()
        self.sub_sample_size = sub_sample_size
        self.rounds = rounds
        self.N = len(self.x)
    
    # 支持"PearsonCorr", "MI-GIEF"两种method参数值
    def _cal_assoc_dists(self, td_lag):
        x_td, y_td = _gen_td_series(self.x, td_lag)
        assocs = _cal_assoc_dist(
            x_td, y_td, self.sub_sample_size, self.rounds)
        x_srg = np.random.permutation(x_td.copy())
        assocs_srg = _cal_assoc_dist(
            x_srg, y_td, self.sub_sample_size, self.rounds)

        return assocs, assocs_srg
        
    def cal_td_assoc_dists(self, td_lags):
        self.td_assocs = []
        self.td_assocs_srg = []
        
        # 逐tau计算
        for i, td_lag in enumerate(td_lags):
            print("%{:.2f}\r".format(i / len(td_lags) * 100), end="")
            assocs, assocs_srg = self._cal_assoc_dists(td_lag)
            
            self.td_assocs.append(assocs)
            self.td_assocs_srg.append(assocs_srg)
        
        return self.td_assocs, self.td_assocs_srg
    

def show_td_analysis_results(td_lags, td_assocs, td_assocs_srg, alpha, show_scatters=True):
    """显示时延自关联分析的结果"""
    bounds = [np.min(td_assocs_srg) - 0.1, np.max(td_assocs) + 0.1]  # 画图的上下边界
    avg_td_cassocs = [np.mean(p) for p in td_assocs]
    ic_ranges = [np.quantile(p, (alpha / 2, 1 - alpha / 2)) for p in td_assocs]
    ic_ranges_srg = [np.quantile(p, (0, 1 - alpha)) for p in td_assocs_srg]

    # 趋势
    plt.plot(td_lags, avg_td_cassocs, "-", color=PROJ_CMAP["blue"], linewidth=1.5, zorder=1)
    plt.fill_between(
        td_lags, [p[0] for p in ic_ranges_srg], [p[1] for p in ic_ranges_srg], alpha=0.3, color=PROJ_CMAP["grey"], zorder=-1)
    plt.fill_between(
        td_lags, [p[0] for p in ic_ranges], [p[1] for p in ic_ranges], alpha=0.3, color=PROJ_CMAP["blue"], zorder=-1)

    # 散点
    if show_scatters:
        for i in range(len(td_assocs[0])):
            plt.scatter(
                td_lags, [p[i] for p in td_assocs], marker="o", s=8, color=PROJ_CMAP["blue"], alpha=0.1, zorder=1)
            plt.scatter(
                td_lags, [p[i] for p in td_assocs_srg], marker="o", s=8, color=PROJ_CMAP["grey"], alpha=0.05, zorder=1)

    plt.ylim(*bounds)
    plt.vlines(0, *bounds, colors="k", linewidth=1.0, zorder=2)

    # 显著性区间
    lag_h = td_lags[1] - td_lags[0]
    signif_td_lags = []
    for idx, td_lag in enumerate(td_lags):
        signif = ic_ranges[idx][0] > ic_ranges_srg[idx][1]
        if signif:
            signif_td_lags.append(td_lag)
    plt.fill_between(
        [min(signif_td_lags) - lag_h / 2, max(signif_td_lags) + lag_h / 2], 
        *bounds, color="red", alpha=0.1, zorder=-1
    )

    plt.grid(alpha=0.3, zorder=-1)