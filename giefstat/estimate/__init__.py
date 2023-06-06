import numpy as np

from ..setting import DTYPES

from .gief import MutualInfoGIEF, CondMutualInfoGIEF, MargEntropy, CondEntropy
from .kde import MutualInfoKDE
from .mic import MIC, CMIC
from .model_based import MutualInfoModel, CondMutualInfoModel
from .quant_based import MutualInfoClassic, MutualInfoDarbellay
from .correlation_coeff import cal_dist_corr, cal_pearson_corr, cal_spearman_corr

ASSOC_METHODS = [
    "PearsonCorr", "SpearmanCorr", "DistCorr",
    "MI-GIEF", "MI-model", "MI-cut", "MI-qcut", "MI-Darbellay", "MI-KDE",
    "MIC", "RMIC"
]
COND_ASSOC_METHODS = [
    "CMI-GIEF", "CMI-model", "CMI-cut", "CMI-qcut",
    "CMIC", "CRMIC"
]


def cal_marg_entropy(x: np.ndarray, xtype: str, **kwargs):
    """计算边际熵

    :param x: x数据, 一维或多维
    :param xtype: x的数值类型, "d"或者"c"
    :param kwargs:
        k: int, 当xtype == "c"时选用的KNN近邻数, defaults to 3
        metric: str, 当xtype == "c"时选用的距离度量方式, defaults to "chebyshev"
    """
    assert xtype in DTYPES
    return MargEntropy(x, xtype)(**kwargs)


def cal_cond_entropy(x: np.ndarray, xtype: str, z: np.ndarray, ztype: str, **kwargs):
    """计算条件熵

    :param x: x数据, 一维或多维
    :param xtype: x的数值类型, "d"或者"c"
    :param z: z数据, 一维或多维
    :param ztype: z的数值类型, "d"或者"c"
    :param kwargs:
        k: int, 当xtype == "c"时选用的KNN近邻数, defaults to 3
        metric: str, 当xtype == "c"时选用的距离度量方式, defaults to "chebyshev"
    """
    assert xtype in DTYPES
    assert ztype in DTYPES
    return CondEntropy(x, xtype, z, ztype)(**kwargs)


def cal_assoc(x, y, method, xtype=None, ytype=None, **kwargs):
    """计算关联系数"""
    assert method in ASSOC_METHODS
    
    # 线性相关系数
    if method == "PearsonCorr":
        return cal_pearson_corr(x, y)
    elif method == "SpearmanCorr":
        return cal_spearman_corr(x, y)
    elif method == "DistCorr":
        return cal_dist_corr(x, y)
    
    # 互信息
    elif method == "MI-GIEF":
        return MutualInfoGIEF(x, xtype, y, ytype)(**kwargs)  # kwargs: k, metric
    elif method == "MI-model":
        return MutualInfoModel(x, xtype, y, ytype)(**kwargs)  # kwargs: model, test_ratio
    elif method == "MI-cut":
        return MutualInfoClassic(x, y)(method="cut")
    elif method == "MI-qcut":
        return MutualInfoClassic(x, y)(method="qcut")
    elif method == "MI-Darbellay":
        return MutualInfoDarbellay(x, y)()
    elif method == "MI-KDE":
        return MutualInfoKDE(x, y)()
    
    # 最大信息系数
    elif method == "MIC":
        return MIC(x, y)(method="mic")
    elif method == "RMIC":
        return MIC(x, y)(method="rmic", **kwargs)
    else:
        raise ValueError(f"unsupported method {method}")
    
    
def cal_cond_assoc(x, y, z, method, xtype=None, ytype=None, ztype=None, **kwargs):
    """计算条件关联系数"""
    assert method in COND_ASSOC_METHODS
    
    # 条件互信息
    if method == "CMI-GIEF":
        return CondMutualInfoGIEF(x, xtype, y, ytype, z, ztype)(**kwargs)  # kwargs: k, metric, method for estimating CMI
    elif method == "CMI-model":
        return CondMutualInfoModel(x, xtype, y, ytype, z, ztype)(**kwargs)  # kwargs: model, test_ratio
    elif method == "CMI-cut":
        return MutualInfoClassic(x, y, z)(method="cut")
    elif method == "CMI-qcut":
        return MutualInfoClassic(x, y, z)(method="qcut")
    
    # 条件最大信息系数
    elif method == "CMIC":
        return CMIC(x, y, z)(method="mic")
    elif method == "CRMIC":
        return CMIC(x, y, z)(method="rmic")
    else:
        raise ValueError(f"unsupported method {method}")