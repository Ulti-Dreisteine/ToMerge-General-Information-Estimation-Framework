import numpy as np

from ..util import exec_model_test


# 紧凑关联识别
def recog_compact(X, y, features_rank, model, n_features_lst, S_pre: set, metric: str="r2", 
                  rounds: int=3):
    """从CMIM-GIEF所得features_rank中识别出y的紧凑关联变量, 返回其在X中的编号表

    :param X: X数组, np.ndarray
    :param y: y数组, np.ndarray
    :param features_rank: CMIM-GIEF所得特征排序表, list
    :param model: 用于进行FIMT测试的机器学习模型
    :param n_features_lst: 用于FIMT测试的特征数列表
    :param S_pre: 在X中已被事先选中的特征集, set
    :param metric: 模型评估指标, str, defaults to "r2"
    :param rounds: 每个特征集合上重复评估的轮数, defaults to 3
    """          

    # 当前仅支持基于决定系数的判断
    if metric != "r2":
        raise ValueError(f"unsupported metric {metric}")

    if S_pre:
        init_value = exec_model_test(X[:, list(S_pre)], y, model, metric=metric, rounds=rounds)[0]
    else:
        init_value = 0.0  # 仅考虑已选特征所得模型分数

    # FIMT测试
    fimt_values = []
    for n_features in n_features_lst:
        S = list(S_pre) + features_rank[:n_features]
        fimt_values.append(exec_model_test(X[:, S], y, model, metric=metric, rounds=rounds)[0])

    # 截断点判断
    max_value = np.max(fimt_values)

    # 如果FIMT测试最大值不高于已选特征集对应的初始值, 则不新选入
    best_n_features = 0
    if max_value - init_value >= 0.01:
        for i, n_features in enumerate(n_features_lst):
            if fimt_values[i] > max_value - 0.01:
                best_n_features = n_features
                break

    # 新识别出的紧凑变量在X所有特征中的序号
    features_compact = features_rank[:best_n_features]
    return features_compact, fimt_values