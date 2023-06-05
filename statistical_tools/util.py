import numpy as np


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