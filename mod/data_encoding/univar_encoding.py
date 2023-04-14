# -*- coding: utf-8 -*-
"""
Created on 2023/04/14 11:13:39

@File -> univar_encoding.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 单变量编码
"""

import warnings

warnings.filterwarnings("ignore")

import category_encoders as ce
import pandas as pd
import numpy as np
import random

EPS = 1e-6

UNSUPER_METHODS = ["ordinal", "random", "count"]
SUPER_METHODS = ["target", "m_estimator", "james_stein", "glmm", "woe", "leave_one_out", "catboost", "mhg"]


class UnsuperCategorEncoding(object):
    """无监督一维类别值编码"""
    
    def __init__(self, x: np.ndarray or list):
        self.x = pd.Series(np.array(x).astype(np.int).flatten(), name = "x")  # type: pd.Series
        self.N = len(x)
        
    def ordinal_encoding(self):
        enc = ce.OrdinalEncoder(cols = ["x"])
        enc.fit(self.x)
        x_enc = enc.transform(self.x)
        return np.array(x_enc).flatten()
    
    def random_encoding(self, seed: int = None):
        # TODO: 优化效率.
        x_sorted = sorted(set(self.x))

        if seed is not None:
            random.seed(seed)
        random.shuffle(x_sorted)

        codes = list(range(1, len(x_sorted) + 1))
        x_enc = self.x.replace(dict(zip(x_sorted, codes)))
        return np.array(x_enc).flatten()
    
    def count_encoding(self):
        # x_enc = self.x.replace(self.x.value_counts().to_dict())
        enc = ce.CountEncoder(cols = ["x"])
        enc.fit(self.x)
        x_enc = enc.transform(self.x)
        return np.array(x_enc)
    
    def encode(self, method: str, **kwargs):
        try:
            assert method in UNSUPER_METHODS
        except:
            raise ValueError(f'Invalid method = \"{method}\"')
        
        x_enc = None
        if method == "ordinal":
            x_enc = self.ordinal_encoding()
        elif method == "random":
            x_enc = self.random_encoding(**kwargs)
        elif method == "count":
            x_enc = self.count_encoding()
        return x_enc
    

class SuperCategorEncoding(object):
    """有监督一维序列编码
    Example
    ----
    super_enc = SuperCategorEncoding(x, y)
    x_enc = super_enc.mhg_encoding()
    """
    
    def __init__(self, x: np.ndarray or list, y: np.ndarray or list):
        """
        初始化
        :param x: x序列, x一定为Nominal类别型变量
        :param y: y序列, y一定为数值型变量
        """
        self.x = pd.Series(np.array(x).astype(np.int).flatten(), name = "x")  # type: pd.Series
        self.y = pd.Series(np.array(y).astype(np.float32).flatten(), name = "y")  # type: pd.Series

    def target_encoding(self):
        enc = ce.TargetEncoder(cols = ["x"], smoothing = 1.0)
        return self._encode_transform(enc)
    
    def m_estimator_encoding(self):
        enc = ce.MEstimateEncoder(cols = ["x"], m = 20.0)
        return self._encode_transform(enc)
    
    def james_stein_encoding(self):
        enc = ce.JamesSteinEncoder(cols = ["x"])
        return self._encode_transform(enc)
    
    def glmm_encoding(self):
        enc = ce.GLMMEncoder(cols = ["x"])
        return self._encode_transform(enc)
    
    def woe_encoding(self, **kwargs):
        y_mean = self.y.mean()
        y_binar = self.y.copy()
        y_binar[np.array(self.y > y_mean).flatten()] = 1
        y_binar[np.array(self.y <= y_mean).flatten()] = 0

        enc = ce.WOEEncoder(cols = ["x"], **kwargs)
        enc.fit(self.x, y_binar)
        x_enc = enc.transform(self.x)
        return np.array(x_enc).flatten()
    
    def leave_one_out_encoding(self):
        enc = ce.LeaveOneOutEncoder(cols = ["x"])
        return self._encode_transform(enc)
    
    def catboost_encoding(self):
        enc = ce.CatBoostEncoder(cols = ["x"])
        return self._encode_transform(enc)

    # TODO Rename this here and in `target_encoding`, `m_estimator_encoding`, `james_stein_encoding`, `glmm_encoding`, `leave_one_out_encoding` and `catboost_encoding`
    def _encode_transform(self, enc):
        """编码并转换"""
        enc.fit(self.x, self.y)
        x_enc = enc.transform(self.x)
        return np.array(x_enc).flatten()
    
    def mhg_encoding(self):
        d = pd.concat([self.x, self.y], axis = 1)
        y_mean = d.groupby("x").mean()
        x_enc = self.x.replace(dict(zip(y_mean.index, list(y_mean["y"]))))
        return np.array(x_enc).flatten()
    
    def encode(self, method: str, **kwargs):
        try:
            assert method in SUPER_METHODS
        except:
            raise ValueError(f'Invalid method = \"{method}\"')
        
        x_enc = None
        if method == "target":
            x_enc = self.target_encoding()
        elif method == "m_estimator":
            x_enc = self.m_estimator_encoding()
        elif method == "james_stein":
            x_enc = self.james_stein_encoding()
        elif method == "glmm":
            x_enc = self.glmm_encoding()
        elif method == "woe":
            x_enc = self.woe_encoding(**kwargs)
        elif method == "leave_one_out":
            x_enc = self.leave_one_out_encoding()
        elif method == "catboost":
            x_enc = self.catboost_encoding()
        elif method == "mhg":
            x_enc = self.mhg_encoding()
        return x_enc