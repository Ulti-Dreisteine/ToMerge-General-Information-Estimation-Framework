# -*- coding: utf-8 -*-
"""
Created on 2023/11/03 09:18:09

@File -> symbolic_te.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: STE
"""

from scipy.signal import find_peaks
from itertools import permutations
import numpy as np
import random


def gen_td_series(x: np.ndarray, y: np.ndarray, td_lag: int):
    x_td_, y_td_ = x.flatten(), y.flatten()
    lag_remain = np.abs(td_lag) % len(x_td_)  # 求余数

    if td_lag == 0:     # 没有时滞, 那么x_td和y_td_1同时发生
        x_td = x_td_[1:].copy()
        y_td = y_td_[1:].copy()
    elif td_lag > 0:    # 正时滞, x_td比y_td_1早lag_remain发生
        x_td = x_td_[:-lag_remain].copy()
        y_td = y_td_[lag_remain:].copy()
    else:               # 负时滞, x_td比y_td_1晚lag_remain发生
        x_td = x_td_[lag_remain + 1:].copy()
        y_td = y_td_[1: -lag_remain].copy()
    return x_td, y_td


def gen_embed_series(x, idxs, m, tau):
    X_embed = x[idxs]
    for i in range(1, m):
        X_embed = np.c_[x[idxs - i * tau], X_embed]
    return X_embed


def continuously_symbolize(x, y, m, tau_x, tau_y):
    """生成具有连续索引的符号样本"""
    patterns = list(permutations(np.arange(m) + 1))
    dict_pattern_index = {patterns[i]: i for i in range(len(patterns))}
    
    idxs = np.arange((m - 1) * max(tau_x, tau_y), len(x))  # 连续索引
    X_embed = gen_embed_series(x, idxs, m, tau_x)
    Y_embed = gen_embed_series(y, idxs, m, tau_y)
    
    X = np.argsort(X_embed) + 1  # NOTE: 滚动形成m维时延嵌入样本  一个时刻成为一个标签
    X = np.array([dict_pattern_index[tuple(p)] for p in X])  # 对应映射到符号上
    
    Y = np.argsort(Y_embed) + 1
    Y = np.array([dict_pattern_index[tuple(p)] for p in Y])
    return X, Y


class SymbolicTransferEntropy(object):
    """
    符号传递熵
    
    example
    -------
    arr = np.load(f"{BASE_DIR}/case_alkaline_hydrogen_te/data/arr.npy")
    cols = load_json_file(f"{BASE_DIR}/case_alkaline_hydrogen_te/file/cols.json")["cols"]
    taus = load_json_file(f"{BASE_DIR}/case_alkaline_hydrogen_te/file/taus.json")
    
    col_x, col_y = "h2_sep_temp", "o2_sep_temp"
    idx_x, idx_y = cols.index(col_x), cols.index(col_y)
    x, y = arr[:, idx_x], arr[:, idx_y]
    tau_x, tau_y = taus[col_x], taus[col_y]
    
    # 符号化
    m = 2
    self = SymbolicTransferEntropy(x, y, tau_x, tau_y, m)
    
    # 延迟样本
    sub_sample_size = None
    rounds = 1
    
    td_lags = np.arange(-50 * tau_x, 51 * tau_x, tau_x)
    td_stes = []
    for i, td_lag in enumerate(td_lags):
        print("%{:.2f}\r".format(i / len(td_lags) * 100), end="")
        ste, _ = self.cal_td_ste(td_lag, sub_sample_size, rounds)
        td_stes.append(ste)
    
    # 背景值
    ste_bg_params = self.cal_bg_ste(sub_sample_size)
    ci_bg_ub = ste_bg_params[0] + 3 * ste_bg_params[1]  # 均值 + 3倍标准差  # 单侧检验
    
    # 峰值信息
    peak_idxs, peak_taus, peak_strengths, peak_signifs = self.parse_peaks(td_lags, td_stes, ci_bg_ub)
    
    # 画图
    bounds = [0, np.max(td_stes) * 1.1]  # 画图的上下边界
    
    # 趋势
    plt.plot(td_lags, td_stes, "-", color=PROJ_CMAP["blue"], linewidth=1.5, zorder=1)
    plt.hlines(ci_bg_ub, td_lags.min(), td_lags.max(), linestyles="--", colors="r")
    plt.ylim(*bounds)
    plt.vlines(0, *bounds, colors="k", linewidth=1.0, zorder=2)
    
    for i, idx in enumerate(peak_idxs):
        if peak_signifs[i]:
            plt.vlines(td_lags[idx], bounds[0], bounds[1], colors=PROJ_CMAP["red"], linewidth=1.0, zorder=5)
    
    plt.grid(alpha=0.3, zorder=-1)
    """
    
    def __init__(self, x: np.ndarray, y: np.ndarray, tau_x: int, tau_y: int, m: int = None) -> None:
        self.x = x.flatten()
        self.y = y.flatten()
        self.tau_x, self.tau_y = tau_x, tau_y
        self.m = 2 if m is None else m
        
        # TODO: 未来改进这块参数
        self.kx, self.ky, self.h = 1, 1, 1

        try:
            self._symbolize()
        except Exception as _:
            raise RuntimeError("Symbolization failed") from _
        
    def _symbolize(self):
        self.X, self.Y = continuously_symbolize(self.x, self.y, self.m, self.tau_x, self.tau_y)
        
    def _cal_ste(self, X, Y, sub_sample_size = None, rounds = None):
        # 计算STE
        idxs = np.arange(0, len(X) - self.h * self.tau_y - 1)
        _Xk = X[idxs].reshape(-1, 1)
        _Yk = Y[idxs].reshape(-1, 1)
        _Yh = Y[idxs].reshape(-1, 1)
        
        concat_XYY = np.concatenate([
            _Xk[self.h * self.tau_y:], 
            _Yk[: -self.h * self.tau_y], 
            _Yh[self.h * self.tau_y:]], axis = 1)
        states_XYY = np.unique(concat_XYY, axis = 0)
        concat_YY = np.concatenate([
            _Yk[: -self.h * self.tau_y], 
            _Yh[self.h * self.tau_y:]], axis = 1)
        
        _N = concat_XYY.shape[0]
        sub_sample_size = _N if sub_sample_size is None else sub_sample_size
        rounds = 10 if rounds is None else rounds
        
        _stes = []
        for _ in range(rounds):
            _idxs = random.sample(range(_N), sub_sample_size)
            _concat_XYY, _concat_YY = concat_XYY[_idxs, :], concat_YY[_idxs, :]
            
            _ste = 0
            eps = 1e-6
            for state in states_XYY:
                prob1 = (_concat_XYY == state).all(axis = 1).sum() / sub_sample_size
                prob2 = (_concat_YY[:, : -1] == state[1 : -1]).all(axis = 1).sum() / sub_sample_size
                prob3 = (_concat_XYY[:, : -1] == state[: -1]).all(axis = 1).sum() / sub_sample_size
                prob4 = (_concat_YY == state[1 :]).all(axis = 1).sum() / sub_sample_size

                prob = prob1 * np.log2((prob1 * prob2) / (prob3 * prob4 + eps) + eps)
                
                if np.isnan(prob) == False:
                    _ste += prob
                    
            _stes.append(_ste)
        
        return np.nanmean(_stes), _stes
        
    def cal_td_ste(self, td_lag, sub_sample_size = None, rounds = None):
        # 时延样本
        X_td, Y_td = gen_td_series(self.X, self.Y, td_lag)
        return self._cal_ste(X_td, Y_td, sub_sample_size, rounds)
    
    @staticmethod
    def _shuffle(x: np.ndarray):
        x_srg = np.random.choice(x, len(x), replace=True)
        return x_srg
    
    def cal_bg_ste(self, sub_sample_size = None, rounds = None):
        """获得背景分布均值和标准差"""
        rounds = 10 if rounds is None else rounds
        _stes_bg = []
        for _ in range(rounds):
            X_shuff, Y_shuff = self._shuffle(self.X), self._shuffle(self.Y)
            _ste_bg, _ = self._cal_ste(X_shuff, Y_shuff, sub_sample_size, rounds=1)
            _stes_bg.append(_ste_bg)
        return np.nanmean(_stes_bg), np.nanstd(_stes_bg)
    
    def parse_peaks(self, td_lags, td_stes, ci_bg_ub, thres = None, distance = None, 
                    prominence = None):
        # 寻找是否有高于阈值的一个或多个峰值, 如果没有则默认峰在0时延处
        # height: 峰值的最小值, distance: 相邻两个峰的最小间距, prominence: 在wlen范围内至少超过最低值的程度
        if thres is None:
            thres = 0.01
            print(f"set default ste thres = {thres}")
        if distance is None:
            distance = len(td_lags) // 10
        if prominence is None:
            prominence = max(td_stes) / 2
        
        peak_idxs, _ = find_peaks(
            td_stes, height=thres, distance=distance, prominence=prominence, 
            wlen=max([2, len(td_lags) // 2]))

        peak_signifs = []
        if len(peak_idxs) == 0:
            peak_taus = []
            peak_strengths = []
        else:
            # 获得对应峰时延、强度和显著性信息
            peak_taus = [td_lags[p] for p in peak_idxs]
            peak_strengths = [td_stes[p] for p in peak_idxs]

            for idx in peak_idxs:
                _n = len(td_lags) // 10
                _series = np.append(td_stes[: _n], td_stes[-_n :])
                _mean, _std = np.mean(_series), np.std(_series)
                signif = (td_stes[idx] > ci_bg_ub) & (td_stes[idx] > _mean + 3 * _std)  # 99% CI
                peak_signifs.append(signif)

        return peak_idxs, peak_taus, peak_strengths, peak_signifs
    
    
    
    
    
    
    