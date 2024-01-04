# -*- coding: utf-8 -*-
"""
Created on 2023/11/30 19:56:54

@File -> adapt_pste.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 使用不同时间尺度的PSTE
"""

from scipy.signal import find_peaks
from typing import List, Tuple
import numpy as np
import random


def gen_td_series(x: np.ndarray, y: np.ndarray, Z: np.ndarray, td_lag: int) -> Tuple[np.ndarray]:
    x_td_, y_td_, Z_td_ = x.flatten(), y.flatten(), Z.copy()
    lag_remain = np.abs(td_lag) % len(x_td_)  # 求余数

    if td_lag == 0:
        # 没有时滞, 那么x_td和y_td_1同时发生
        x_td = x_td_[1:].copy()
        y_td = y_td_[1:].copy()
        Z_td = Z_td_[1:, :].copy()
    elif td_lag > 0:
        # 正时滞, x_td比y_td_1早lag_remain发生
        x_td = x_td_[:-lag_remain].copy()
        y_td = y_td_[lag_remain:].copy()
        Z_td = Z_td_[lag_remain:, :].copy()
    else:
        # 负时滞, x_td比y_td_1晚lag_remain发生
        x_td = x_td_[lag_remain + 1:].copy()
        y_td = y_td_[1: -lag_remain].copy()
        Z_td = Z_td_[1: -lag_remain, :].copy()
    return x_td, y_td, Z_td


class PartialTransferEntropy():
    """偏传递熵"""
    
    def __init__(self, x: np.ndarray, y: np.ndarray, Z: np.ndarray, tau_x: int, tau_y: int, 
                 sub_sample_size: int = None, alpha: float = 0.01, rounds: int = 50):
        # NOTE: x、y和Z的元素都必须为整数值
        self.x = x.flatten()
        self.y = y.flatten()
        self.Z = Z.reshape(len(Z), -1).copy()
        self.tau_x = tau_x
        self.tau_y = tau_y
        self.sub_sample_size = sub_sample_size
        self.alpha = alpha
        self.rounds = rounds
        self.N = len(self.x)
        
        # TODO: 未来改进这块参数
        self.kx, self.ky, self.h = 1, 1, 1
    
    def cal_td_te(self, td_lag, sub_sample_size = None, rounds = None):
        # 时延样本
        x_td, y_td, Z_td = gen_td_series(self.x, self.y, self.Z, td_lag)
        return self.cal_te(x_td, y_td, Z_td, sub_sample_size, rounds) 
    
    def cal_te(self, x, y, Z, sub_sample_size = None, rounds = None):
        idxs = np.arange(0, len(x) - self.h * self.tau_y - 1)
        _xk = x[idxs].reshape(-1, 1)
        _Zk = Z[idxs, :]
        _yk = y[idxs].reshape(-1, 1)
        _yh = y[idxs].reshape(-1, 1)
        
        concat_xzyy = np.concatenate([
            _xk[self.h * self.tau_y:], 
            _Zk[self.h * self.tau_y:], 
            _yk[: -self.h * self.tau_y],
            _yh[self.h * self.tau_y:], 
            ], axis=1)
        states_xzyy = np.unique(concat_xzyy, axis = 0)
        
        _N = concat_xzyy.shape[0]
        
        sub_sample_size = _N if sub_sample_size is None else sub_sample_size
        rounds = 10 if rounds is None else rounds
        
        te_rounds = []
        for _ in range(rounds):
            _idxs = random.sample(range(_N), sub_sample_size)
            _concat_xzyy = concat_xzyy[_idxs, :]
            
            _te = 0
            eps = 1e-6
            for state in states_xzyy:
                # P(xk, zk, yk, yh)
                prob1 = (_concat_xzyy == state).all(axis = 1).sum() / sub_sample_size
                
                # P(zk, yk)
                prob2 = (_concat_xzyy[:, 1 : -1] == state[1 : -1]).all(axis = 1).sum() / sub_sample_size
                
                # P(xk, zk, yk)
                prob3 = (_concat_xzyy[:, : -1] == state[: -1]).all(axis = 1).sum() / sub_sample_size
                
                # p(zk, yk, yh)
                prob4 = (_concat_xzyy[:, 1 :] == state[1 :]).all(axis = 1).sum() / sub_sample_size

                prob = prob1 * np.log2((prob1 * prob2) / (prob3 * prob4 + eps) + eps)
                
                if np.isnan(prob) == False:
                    _te += prob
                    
            te_rounds.append(_te)
        
        return np.nanmean(te_rounds), np.nanstd(te_rounds), te_rounds

    @staticmethod
    def _shuffle(x: np.ndarray):
        idxs = np.arange(len(x))
        idxs_shuff = np.random.choice(idxs, len(x), replace=True)
        
        if len(x.shape) == 1:
            x_srg = x[idxs_shuff]
        elif len(x.shape) == 2:
            x_srg = x[idxs_shuff, :]
        else:
            raise RuntimeError
        
        return x_srg
    
    def cal_bg_te(self, sub_sample_size = None, rounds = None):
        """获得背景分布均值和标准差"""
        rounds = 10 if rounds is None else rounds
        _tes_bg = []
        for _ in range(rounds):
            x_shuff, y_shuff, Z_shuff = self._shuffle(self.x), self._shuffle(self.y), self._shuffle(self.Z)
            _te_bg, _, _ = self.cal_te(x_shuff, y_shuff, Z_shuff, sub_sample_size, rounds=1)
            _tes_bg.append(_te_bg)
        return np.nanmean(_tes_bg), np.nanstd(_tes_bg)  # TODO: 返回结果最后加上_tes_bg, 与self.cal_te()保持一致
    
    def parse_peaks(self, td_lags: np.ndarray, td_tes: List[tuple], ci_bg_ub: float, 
                    thres: float = None, distance: int = None, prominence: float = None):
        # 寻找是否有高于阈值的一个或多个峰值, 如果没有则默认峰在0时延处
        # height: 峰值的最小值, distance: 相邻两个峰的最小间距, prominence: 在wlen范围内至少超过最低值的程度
        if thres is None:
            thres = ci_bg_ub
            # print(f"set default te thres = {thres}")
        if distance is None:
            distance = 2 * self.tau_x
        if prominence is None:
            prominence = 0.05
        
        td_te_means = [p[0] for p in td_tes]
        td_te_stds = [p[1] for p in td_tes]
        
        peak_idxs, _ = find_peaks(
            td_te_means, height=thres, distance=distance, prominence=prominence, #width=width,
            wlen=max([2, len(td_lags) // 2]),
            )

        peak_signifs = []
        if len(peak_idxs) == 0:
            peak_taus = []
            peak_strengths = []
            peak_stds = []
        else:
            # 获得对应峰时延、强度和显著性信息
            peak_taus = [td_lags[p] for p in peak_idxs]
            peak_strengths = [td_te_means[p] for p in peak_idxs]
            peak_stds = [td_te_stds[p] for p in peak_idxs]

            for idx in peak_idxs:
                _n = len(td_lags) // 10
                _series = np.append(td_te_means[: _n], td_te_means[-_n :])
                _mean, _std = np.mean(_series), np.std(_series)
                signif = (td_te_means[idx] > ci_bg_ub) & (td_te_means[idx] > _mean + 3 * _std)  # 99% CI
                peak_signifs.append(signif)

        return peak_idxs, peak_taus, peak_strengths, peak_stds, peak_signifs