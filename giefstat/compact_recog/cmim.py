# sourcery skip: avoid-builtin-shadow
from joblib import Parallel, delayed
from typing import List
import numpy as np
import random

from ..estimate import cal_assoc, cal_cond_assoc
from ..util import stdize_values

EPS = 1e-12
MI_METHOD = "MI-GIEF"
CMI_METHOD = "CMI-GIEF"

__doc__ = """
    基于CMIM-GIEF的紧凑关联变量识别:
    NOTE:
    1. 在计算中将所有变量视为连续类型, 经过stdize_value处理为标准格式
"""


# ---- 关联系数计算 ---------------------------------------------------------------------------------

def cal_mi(x, y, k, n):
    """计算I(x;y)"""
    x, y = x.flatten(), y.flatten()
    idxs = random.sample(range(x.shape[0]), k=n)    # 随机无返回抽样
    x, y = x[idxs], y[idxs]
    return cal_assoc(x, y, MI_METHOD, "c", "c", k=k)


def cal_cmi(xi, xs, y, k, n):
    """计算I(xi;y|xs)"""
    xi, xs, y = xi.flatten(), xs.flatten(), y.flatten()
    idxs = random.sample(range(xi.shape[0]), k=n)   # 随机无返回抽样
    xi, xs, y = xi[idxs], xs[idxs], y[idxs]
    return cal_cond_assoc(xi, y, xs, CMI_METHOD, "c", "c", "c", k=k)


# ---- CMIM-GIEF -----------------------------------------------------------------------------------

class CondMIMaximization(object):
    """条件关联系数最大化"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, n_jobs: int = 2, k: int = 3, n: int=500):
        """初始化"""
        # 将所有变量均视为连续值类型, 加入噪声并归一化
        self.X = stdize_values(X, "c", EPS)
        self.y = stdize_values(y, "c", EPS)
        self.N, self.D = self.X.shape
        self.n_jobs = n_jobs
        self.k = k      # MI-GIEF中计算的K近邻数
        self.n = n      # 用于限制MI-GIEF和CMI-GIEF计算样本量
        
    def _cal_mi(self, x, y):
        return cal_mi(x, y, self.k, self.n)
    
    def _cal_cmi(self, xi, xs, y):
        return cal_cmi(xi, xs, y, self.k, self.n)
        
    def _init_assocs(self) -> List[float]:
        """计算所有特征与目标的关联系数"""
        with Parallel(n_jobs=self.n_jobs) as parallel:
            assocs = parallel(delayed(self._cal_mi)(self.X[:, i], self.y) for i in range(self.D))
        return assocs
    
    def _recog_redundants(self, S: set, F: set, cmi_thres=0.1) -> set:
        R_new = set()
        for fi in F.difference(S):
            for fs in S:
                cassoc = self._cal_cmi(self.X[:, fi], self.X[:, fs], self.y)
                if cassoc < cmi_thres:
                    R_new.update({fi})
                    break
        return R_new
    
    def feature_selection(self, S_pre: set = None, max_iters: int = 30, verbose = False):
        """特征选择
        :param S_pre: 指定X中已被提前选中的特征集合, defaults to None
        :param max_iters: 迭代选择的最大次数, defaults to 30
        :param verbose: 是否打印过程输出, defaults to False
        """
        F = set(range(self.D))  # 全量特征集合
        S = set() if S_pre is None else S_pre.copy()
        R = set()
        features_rank = []  # 用于记录所选特征顺序

        # 识别冗余特征集R
        if S:
            R.update(self._recog_redundants(S, F))

        cmi_matrix = None                   # 用于记录迭代中的CMI值
        min_cmi = np.ones(len(F)) * np.nan  # 用于记录迭代中的最小CMI值
        i = 0
        while True:
            C = F.difference(S).difference(R)  # 剩余候选特征

            if (not C) or (len(features_rank) > max_iters):
                break
            if not S:  # 如果一个被选的特征都没有, 则采用单因素分析中所得选择关联度最强的
                self.assocs = self._init_assocs()
                f = np.argmax(self.assocs)
            else:
                if cmi_matrix is None:
                    cmi_matrix = np.ones((len(F), len(F))) * np.nan
                    for fi in C:
                        xi = self.X[:, fi]
                        for fs in S:
                            xs = self.X[:, fs]
                            cmi_matrix[fi, fs] = self._cal_cmi(xi, xs, self.y)
                        min_cmi[fi] = np.nanmin(cmi_matrix[fi, :])
                else:
                    xf = self.X[:, f]
                    for fi in C:
                        xi = self.X[:, fi]
                        cmi_matrix[fi, f] = self._cal_cmi(xi, xf, self.y)
                        min_cmi[fi] = np.nanmin([min_cmi[fi], cmi_matrix[fi, f]])

                f = list(C)[np.nanargmax(min_cmi[list(C)])]
            
            R_new = self._recog_redundants({f}, F.difference(S))
            S.update({f})           # 加入新选择的特征f
            R.update(R_new)         # 增加候选特征中关于f的冗余
            features_rank.append(f)
            
            if verbose:
                print(f"iteration i={i}: select feature no. {f}")
            
            i += 1
            if i == max_iters:
                break
            
        details = (F, S, R, cmi_matrix, min_cmi)
        return features_rank, details