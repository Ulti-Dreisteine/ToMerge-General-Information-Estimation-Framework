# -*- coding: utf-8 -*-
"""
Created on 2023/06/09 11:01:37

@File -> incremental_cmim.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 使用CMIM进行增量特征选择
"""

from joblib import Parallel, delayed
import numpy as np
from typing import List
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
sys.path.insert(0, BASE_DIR)

from giefstat.estimate import cal_assoc, cal_cond_assoc
from giefstat.util import normalize

EPS = 1e-12
MI_METHOD = "MI-GIEF"
CMI_METHOD = "CMI-GIEF"

# TODOs:
# 1. 检查不同类型变量间使用cal_assoc和cal_cond_assoc的通用性


class CondMIMaximization(object):
    """条件关联系数最大化"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, n_jobs: int = 2, k: int = 3, ns: int=500):
        """初始化"""
        self.X = normalize(X.reshape(X.shape[0], -1))
        self.y = normalize(y.reshape(-1, 1)).flatten()
        self.N, self.D = self.X.shape
        self.n_jobs = n_jobs
        self.k = k
        self.ns = ns

        self.X += np.random.random(self.X.shape) * EPS
        self.y += np.random.random(self.y.shape) * EPS
        
    def _init_assocs(self) -> List[float]:
        """计算所有特征与目标的关联系数"""
        with Parallel(n_jobs=self.n_jobs) as parallel:
            # FIXME: fix code here
            assocs = parallel(delayed(cal_assoc)(
                self.X[:, i], self.y, k=self.k, n=self.ns) for i in range(self.D))
        return assocs
    
    def _recog_redundants(self, S: set, F: set, cmi_thres=0.1) -> set:
        R_new = set()
        for fi in F.difference(S):
            for fs in S:
                cassoc = cal_cond_assoc(
                    self.X[:, fi], self.X[:, fs], self.y, CMI_METHOD, "c", "c", "c", k=self.k)
                if cassoc < cmi_thres:
                    R_new.update({fi})
                    break
        return R_new
    
    # TODO: 优化以下部分
    # <<<<<<<<
    def feature_selection(self, S: set = None, iter_n: int = 30, verbose = False):
        """特征选择
        :param S: 指定X中已被提前选中的特征集合, defaults to None
        :param iter_n: 迭代选择的最大次数, defaults to 30
        :param verbose: 是否打印过程输出, defaults to False
        """
        F = set(range(self.X.shape[1]))  # 全量特征集合
        S = set() if S is None else S
        R = set()
        records = []  # 用于记录选择进程

        if len(S) > 0:
            # 识别冗余特征集R
            R.update(self._recog_redundants(S, F))

        cond_assocs_matrix = None
        min_cond_assocs = np.ones(len(F)) * np.nan
        i = 0
        while True:
            C = F.difference(S).difference(R)  # 剩余候选特征

            if len(C) == 0:
                break
            elif len(records) > iter_n:
                break

            if len(S) == 0:  # 如果一个被选的特征都没有, 则采用单因素分析中所得选择关联度最强的
                self.assocs = self._init_assocs()
                f = np.argmax(self.assocs)
            else:
                if cond_assocs_matrix is None:
                    cond_assocs_matrix = np.ones((len(F), len(F))) * np.nan
                    for fi in C:
                        xi = self.X[:, fi]
                        for fs in S:
                            xs = self.X[:, fs]
                            cond_assocs_matrix[fi, fs] = cal_cond_assoc(
                                xi, xs, self.y, k=self.k, n=self.ns)
                        min_cond_assocs[fi] = np.nanmin(
                            cond_assocs_matrix[fi, :])
                else:
                    xf = self.X[:, f]
                    for fi in C:
                        xi = self.X[:, fi]
                        cond_assocs_matrix[fi, f] = cal_cond_assoc(
                            xi, xf, self.y, k=self.k, n=self.ns)
                        min_cond_assocs[fi] = np.nanmin(
                            [min_cond_assocs[fi], cond_assocs_matrix[fi, f]])

                f = list(C)[np.nanargmax(min_cond_assocs[list(C)])]

            R_new = self._recog_redundants({f}, F.difference(S))
            S.update({f})  # 加入新选择的特征f
            R.update(R_new)  # 增加候选特征中关于f的冗余
            records.append(f)
            
            if verbose:
                print("select %d" % f)
            
            i += 1
            if i == iter_n:
                break

        details = (F, S, R, cond_assocs_matrix, min_cond_assocs)
        return records, details
    # >>>>>>>>