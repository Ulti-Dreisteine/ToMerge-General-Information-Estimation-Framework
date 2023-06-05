# -*- coding: utf-8 -*-
# sourcery skip: avoid-builtin-shadow
"""
Created on 2022/09/18 17:04:28

@File -> mi_cmi.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 基于模型估计MI和CMI
"""

__doc__ = """
    使用模型近似估计互信息, 原理如下
    I(x;Y) = Accu(y, model.predict(x))
    I(x;Y|Z) = I(Y; xZ) - I(Y; Z)
             = Accu(y, model.predict(xz)) - Accu(y, model.predict(z))
"""

from sklearn.metrics import f1_score as cal_f1, r2_score as cal_r2
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
sys.path.insert(0, BASE_DIR)

from estimate.setting import DTYPES
from estimate.util import stdize_values


def _cal_metric(y_true, y_pred, ytype):
    if ytype == "d":
        return cal_f1(y_true, y_pred, average="micro")  # note 使用微平均
    else:
        return cal_r2(y_true, y_pred)


def _exec_model_test(x, y, ytype, model, test_ratio, z=None):
    N = x.shape[0]
    test_size = int(N * test_ratio)
    shuffled_indexes = np.random.permutation(range(N))
    train_idxs = shuffled_indexes[test_size:]
    test_idxs = shuffled_indexes[:test_size]

    x_train, x_test = x[train_idxs, :], x[test_idxs, :]
    y_train, y_test = y[train_idxs], y[test_idxs]

    model.fit(x_train, y_train)
    accu_x = _cal_metric(y_test, model.predict(x_test), ytype)

    if z is None:
        return accu_x
    
    xz = np.c_[x, z]
    xz_train, xz_test = xz[train_idxs, :], xz[test_idxs, :]
    model.fit(xz_train, y_train)
    accu_xz = _cal_metric(y_test, model.predict(xz_test), ytype)
    return accu_xz - accu_x


class MutualInfoModel(object):
    """基于模型的MI"""
    
    def __init__(self, x: np.ndarray, xtype: str, y: np.ndarray, ytype: str):
        assert xtype in DTYPES
        assert ytype in DTYPES
        self.x_norm = stdize_values(x, xtype)
        self.y_norm = stdize_values(y, ytype)
        self.xtype = xtype
        self.ytype = ytype
        
    def __call__(self, model, test_ratio=0.3):
        return _exec_model_test(
            self.x_norm, self.y_norm.flatten(), self.ytype, model, test_ratio)
        
        
class CondMutualInfoModel(object):
    """基于模型的CMI"""
    
    def __init__(self, x: np.ndarray, xtype: str, y: np.ndarray, ytype: str, z: np.ndarray, ztype: str):
        assert xtype in DTYPES
        assert ytype in DTYPES
        assert ztype in DTYPES
        self.x_norm = stdize_values(x, xtype)
        self.y_norm = stdize_values(y, ytype)
        self.z_norm = stdize_values(z, ztype)
        self.xtype = xtype
        self.ytype = ytype
        self.ztype = ztype
        
    def __call__(self, model, test_ratio=0.3):
        return _exec_model_test(
            self.x_norm, self.y_norm.flatten(), self.ytype, model, test_ratio, self.z_norm)
        

if __name__ == "__main__":
    def test_mi():
        x = np.random.normal(0, 1, 10000)
        y = np.random.randint(0, 5, 10000)
        
        self = MutualInfoModel(x, "c", y, "d")
    
        from lightgbm import LGBMClassifier as LightGBM
        model = LightGBM(
            n_estimators=100,
            max_depth=3,
            boosting_type="goss",
            learning_rate=0.1,
            importance_type="split",
            n_jobs=3,
        )
    
        print(f"mi = {self(model)}")
    
    def test_cmi():
        x = np.random.normal(0, 1, 10000)
        y = np.random.normal(0, 1, 10000)
        z = np.random.normal(0, 1, 10000)
        
        self = CondMutualInfoModel(x, "c", y, "c", z, "c")

        from lightgbm import LGBMRegressor as LightGBM
        model = LightGBM(
            n_estimators=100,
            max_depth=3,
            boosting_type="goss",
            learning_rate=0.1,
            importance_type="split",
            n_jobs=3,
        )

        print(f"cmi = {self(model)}")
        
    test_mi()  # todo 分类问题算得关联值下限不为0
    test_cmi()