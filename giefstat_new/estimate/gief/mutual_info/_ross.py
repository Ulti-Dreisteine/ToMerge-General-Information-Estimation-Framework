from scipy.special import psi
from numpy import log
import numpy as np

from ....util import build_tree, query_neighbors_dist


# ---- 高维数据压缩, 计算互信息时要用. ----------------------------------------------------------------

def _reorder_z_series(z_compress: np.ndarray):
    z_compress = z_compress.copy()
    map_ = dict(zip(set(z_compress), list(range(len(set(z_compress))))))
    vfunc = np.vectorize(lambda x: map_[x])
    z_compress = vfunc(z_compress)
    return z_compress


def compress_z_data(z_arr: np.ndarray):
    """对高维数据进行压缩
    :param z_arr: Z数组, 注意shape = (D, N)而不是(N, D)
    """
    D = z_arr.shape[0]
    z_label_ns = [len(set(z_arr[i, :])) for i in range(z_arr.shape[0])]

    if D == 0:
        raise ValueError('empty array z')

    if D == 1:
        z_compress = z_arr.flatten()
        z_compress = _reorder_z_series(z_compress)
    else:
        i = 2
        z_compress = np.ravel_multi_index(z_arr[:i, :], z_label_ns[:i], mode='wrap')
        z_compress = _reorder_z_series(z_compress)

        while i != D:
            _arr = np.vstack((z_compress, z_arr[i, :]))
            z_compress = np.ravel_multi_index(_arr, (len(set(z_compress)), z_label_ns[i]), mode='wrap')
            z_compress = _reorder_z_series(z_compress)
            i += 1

    return z_compress


def cal_ross_mi(x, y, k, metric="chebyshev", base=np.e):
    x, y = x.copy(), y.copy()
    N = x.shape[0]
    
    # Note: 对高维y进行重编码, 压缩为1维序列
    y = y.reshape(y.shape[0], -1)
    if y.shape[1] > 1:
        y = compress_z_data(y.T).reshape(-1, 1)
        
    assert (x.shape[1] >= 1) & (y.shape[1] == 1)  # NOTE: 此处y必须为1维
    y = y.flatten()

    k = 3 if k is None else k
    assert k <= N - 1

    # 统计各类Y的总数.
    classes = np.unique(y)
    Nx_class = np.zeros_like(y)
    for i, _ in enumerate(y):
        Nx_class[i] = np.sum(y == y[i])

    # 逐类进行K近邻计算.
    nn_distc_classes = np.zeros_like(y, dtype=float)
    for c in classes:
        mask = np.where(y == c)[0]
        tree = build_tree(x[mask, :], "chebyshev")
        nn_distc_classes[mask] = query_neighbors_dist(
            tree, x[mask, :], k)  # 获得了各样本第k近邻的距离

    # 所有样本中的K近邻计算.
    tree = build_tree(x, metric)
    m = tree.query_radius(x, nn_distc_classes)
    m = [p.shape[0] for p in m]
    return (psi(N) - np.mean(psi(Nx_class)) + psi(k) - np.mean(psi(m))) / log(base)