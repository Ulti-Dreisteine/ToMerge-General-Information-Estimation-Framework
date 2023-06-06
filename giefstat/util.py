import warnings

warnings.filterwarnings("ignore")

from sklearn.neighbors import BallTree, KDTree
from sklearn.preprocessing import MinMaxScaler
from scipy.special import gamma
import category_encoders as ce
import pandas as pd
import numpy as np
import random


# #### 数据编码 #####################################################################################

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
        # NOTE x值需为int
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
    

# #### 数据处理 #####################################################################################


# ---- 数据标准化 -----------------------------------------------------------------------------------

def normalize(X):
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X.copy())
    return X


def _convert_1d_series2int(x):
    """将一维数据按照标签进行编码为连续整数"""
    x = x.flatten()
    x_unique = np.unique(x)

    if len(x_unique) > 100:
        raise RuntimeWarning(
            f"too many labels: {len(x_unique)} for the discrete data")

    x = np.apply_along_axis(lambda x: np.where(
        x_unique == x)[0][0], 1, x.reshape(-1, 1))
    return x


def _convert_arr2int(arr):
    """将一维数据按照标签进行编码为连续整数"""
    _, D = arr.shape
    for d in range(D):
        arr[:, d] = _convert_1d_series2int(arr[:, d])
    return arr.astype(int)


def stdize_values(x: np.ndarray, dtype: str, eps: float = 1e-10):
    """数据预处理: 标签值整数化、连续值归一化, 将连续和离散变量样本处理为对应的标准格式用于后续分析"""
    x = x.copy()
    x = x.reshape(x.shape[0], -1)
    if dtype == "c":
        # 连续值加入噪音并归一化
        x += eps * np.random.random_sample(x.shape)
        return normalize(x)
    elif dtype == "d":
        # 将标签值转为连续的整数值
        x = _convert_arr2int(x)
        return x
    

# ---- 数据离散化 -----------------------------------------------------------------------------------

def _discretize_series(x: np.ndarray, n: int = 30, method="qcut"):
    """对数据序列采用等频分箱"""
    q = int(len(x) // n)
    if method == "qcut":
        x_enc = pd.qcut(x, q, labels=False, duplicates="drop").flatten()  # 等频分箱
    elif method == "cut":
        x_enc = pd.cut(x, q, labels=False, duplicates="drop").flatten()  # 等宽分箱
    return x_enc


def discretize_arr(X: np.ndarray, n: int = None, method: str = "qcut"):
    """逐列离散化"""
    if n is None:
        n = X.shape[0] // 20
    X = X.copy()
    for i in range(X.shape[1]):
        X[:, i] = _discretize_series(X[:, i], n, method)
    return X.astype(int)
    

# ---- K近邻查询 ------------------------------------------------------------------------------------
    
def build_tree(x, metric: str = "chebyshev"):
    """建立近邻查询树. 低维用具有欧式距离特性的KDTree; 高维用具有更一般距离特性的BallTree"""
    return BallTree(x, metric=metric) if x.shape[1] >= 20 else KDTree(x, metric=metric)


def query_neighbors_dist(tree: BallTree or KDTree, x, k: int):
    """求得x样本在tree上的第k个近邻样本"""
    return tree.query(x, k=k + 1)[0][:, -1]


# ---- 空间球体积 -----------------------------------------------------------------------------------

def get_unit_ball_volume(d: int, metric: str = "euclidean"):
    """d维空间中按照euclidean或chebyshev距离计算所得的单位球体积"""
    if metric == "euclidean":
        return (np.pi ** (d / 2)) / gamma(1 + d / 2)  
    elif metric == "chebyshev":
        return 1
    else:
        raise ValueError(f"unsupported metric {metric}")
    

# ---- 构造时延序列 ---------------------------------------------------------------------------------

def build_td_series(x: np.ndarray, y: np.ndarray, tau: int, max_len: int = 5000):
    """
    生成时延序列样本
    :param x: 样本x数组
    :param y: 样本y数组
    :param tau: 时间平移样本点数, 若 tau > 0, 则 x 对应右方 tau 个样本点后的 y; 
        若 tau < 0, 则 y 对应右方 tau 个样本点后的 x
    """
    x_td, y_td = x.flatten(), y.flatten()
    
    if len(x_td) != len(y_td):
        raise ValueError("length of x is not equal to y")
    
    N = len(x_td)
    
    lag_remain = np.abs(tau) % N
    if lag_remain != 0:
        if tau > 0:
            y_td = y_td[lag_remain:]
            x_td = x_td[:-lag_remain]
        else:
            x_td = x_td[lag_remain:]
            y_td = y_td[:-lag_remain]

    # 当样本量过高时执行降采样, 降低计算复杂度
    if len(x_td) > max_len:
        idxs = np.arange(len(x_td))
        np.random.shuffle(idxs)
        idxs = idxs[:max_len]
        x_td, y_td = x_td[idxs], y_td[idxs]

    return x_td, y_td