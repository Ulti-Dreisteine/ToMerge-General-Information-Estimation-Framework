# -*- coding: utf-8 -*-
"""
Created on 2022/09/29 10:29:51

@File -> fcbf_pc_search.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 基于FCBF的PC节点快速搜索算法
"""

from pyitlib import discrete_random_variable as drv
import pandas as pd
import numpy as np


# 数据离散化
def discretize_series(x: np.ndarray, q = 15, method = "qcut"):
    """对数据序列采用等频分箱"""
    # 分箱方式.
    if method == "qcut":
        x_enc = pd.qcut(x.flatten(), q, labels=False, duplicates="drop").flatten()  # 等频分箱
    elif method == "cut":
        x_enc = pd.cut(x.flatten(), q, labels=False, duplicates="drop").flatten()  # 等宽分箱
    else:
        raise RuntimeError("Unknown method")
        
    return x_enc


def _cal_su(x: np.ndarray, y: np.ndarray, q, encode = True, eps = 1e-12):
    """计算SU值"""
    if encode:
        x_enc = discretize_series(x, q=q, method="qcut").astype(int)
        y_enc = discretize_series(y, q=q, method="qcut").astype(int)
        x_enc, y_enc = x_enc.flatten(), y_enc.flatten()
    else:
        x_enc, y_enc = x.flatten(), y.flatten()
    mi = drv.information_mutual(x_enc, y_enc)
    Hx = drv.entropy(x_enc)
    Hy = drv.entropy(y_enc)
    # return 2 * mi / (Hx + Hy + eps)
    return 2 * mi / (Hx + Hy)


# note 创新点
def surrog_compare(xi, xj, y, q, sub_sample_size: int, rounds=30, encode=True):
    diffs = np.array([])
    for _ in range(rounds):
        idxs = np.random.permutation(np.arange(len(xi)))[:sub_sample_size]
        xi_, xj_, y_ = xi[idxs], xj[idxs], y[idxs]
        diff = _cal_su(xi_, xj_, q, encode=encode) > _cal_su(y_, xj_, q, encode=encode)
        diffs = np.append(diffs, diff)
    p = len(diffs[diffs < 0]) / rounds  # p值
    return p
    

class PCSearch(object):
    """基于FCBF的PC集快速筛选"""
    
    def __init__(self, D: np.ndarray, delta=0.05, alpha=0.01, encode=True):
        """初始化"""
        self.D = D.copy()
        self.nodes = np.arange(D.shape[1], dtype=int)
        self.delta = delta
        self.alpha = alpha
        self.encode = encode  # 参数提醒
    
    def get_pc_lst(self, y_node, repeats=1, sub_sample_size=1000) -> list:
        """从数据集D中获得y_node节点的PC节点集, 返回集合按照重要度递减排序"""
        x_nodes = [p for p in self.nodes if p != y_node]
        
        y = self.D[:, y_node].flatten()
        X = self.D[:, x_nodes]
        
        # 从X中过滤与y有关联的节点
        Sx, sus = self._filter_nodes(X, y)

        # 重排序
        Sx, _ = self._resort(Sx, sus)
        
        if len(Sx) == 0:
            return []
        
        Sx_pc_mrg = None
        for _ in range(repeats):
            idxs = np.random.permutation(np.arange(len(X)))[:sub_sample_size]
            X_, y_ = X[idxs, :], y[idxs]
            Sx_pc = self._get_inner_pc(X_, y_, Sx, self.encode)
            
            if Sx_pc_mrg is None:
                Sx_pc_mrg = Sx_pc
            else:
                Sx_pc_mrg = [p for p in list(Sx_pc_mrg) if p in Sx_pc]
            
        return [x_nodes[p] for p in Sx_pc_mrg]  # 返回有序列表
        
    def _filter_nodes(self, X, y, q = 15):
        """按照固定delta值从X中筛选节点, 并记录对应SU值"""
        S, sus = np.array([]), np.array([])
        for idx_i in range(X.shape[1]):  # note 这里S和sus都是以X为准
            x = X[:, idx_i]
            su = _cal_su(x, y, q, encode=self.encode)
            # print(su)
            if su > self.delta:
                S = np.append(S, idx_i)  # note S为X中的序号
                sus = np.append(sus, su)
        return S, sus

    @staticmethod
    def _resort(S, sus):
        """将S按照su值降序排列"""
        idxs_sort = np.argsort(sus)[::-1]
        S = S[idxs_sort].astype(int)
        sus = sus[idxs_sort]
        return S, sus
    
    @staticmethod
    def _get_inner_pc(X, y, Sx, encode, q = 15, alpha = 0.01):
        """从X数组中选出y的PC集合, 注意S是X中与Y有SU关系的编码, len(S) != X.shape[1]"""
        S_mask = np.ones_like(Sx)
        idx_i = 0
        while True:
            if S_mask[idx_i] != 0:
                xi = X[:, Sx[idx_i]]
                for idx_j in range(idx_i + 1, len(Sx)):
                    xj = X[:, Sx[idx_j]]
                    if S_mask[idx_j] == 0:
                        continue
                    
                    # <<<< 基于显著性检验
                    # elif surrog_compare(xi, xj, y, q, encode=encode, sub_sample_size=1000) < alpha:  # todo 设置阈值
                        # S_mask[idx_j] = 0
                    # <<<< 基于阈值筛选
                    elif _cal_su(xi, xj, q, encode=encode) - _cal_su(y, xj, q, encode=encode) > 0.0:  # todo 设置该阈值
                        S_mask[idx_j] = 0

            idx_i += 1

            # 终止条件
            if idx_i == len(Sx):
                break

        return Sx[S_mask==1]