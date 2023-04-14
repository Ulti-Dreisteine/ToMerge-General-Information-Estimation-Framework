from sklearn.preprocessing import MinMaxScaler
from scipy.special import gamma
from sklearn.neighbors import BallTree, KDTree
import pandas as pd
import numpy as np


# ---- 数据标准化处理 --------------------------------------------------------------------------------

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
        # see: https://en.wikipedia.org/wiki/Volume_of_an_n-ball.
        return (np.pi ** (d / 2)) / gamma(1 + d / 2)  
    elif metric == "chebyshev":
        return 1
    else:
        raise ValueError(f"unsupported metric {metric}")