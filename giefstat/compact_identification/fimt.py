from sklearn.metrics import mean_squared_error as mse, explained_variance_score as evs,\
    r2_score as r2
import numpy as np


def _cal_metric(y_true, y_pred, metric: str):
    y_true, y_pred = y_true.flatten(), y_pred.flatten()
    if metric == "r2":
        return r2(y_true, y_pred)
    if metric == "evs":
        return evs(y_true, y_pred)
    if metric == "mse":
        return mse(y_true, y_pred)
    if metric == "mape":
        idxs = np.where(y_true != 0)
        y_true = y_true[idxs]
        y_pred = y_pred[idxs]
        return np.sum(np.abs((y_pred - y_true) / y_true)) / len(y_true)
    

def _train_test_split(X, y, seed: int = None, test_ratio=0.3):
    # sourcery skip: hoist-similar-statement-from-if, hoist-statement-from-if
    X, y = X.copy(), y.flatten()
    assert X.shape[0] == y.shape[0]
    assert 0 <= test_ratio < 1

    if seed is not None:
        np.random.seed(seed)
        shuffled_indexes = np.random.permutation(range(len(X)))
    else:
        shuffled_indexes = np.random.permutation(range(len(X)))
    test_size = int(len(X) * test_ratio)
    train_index = shuffled_indexes[test_size:]
    test_index = shuffled_indexes[:test_size]
    return X[train_index, :], X[test_index, :], y[train_index], y[test_index]


def eval_prediction_effect(X, y, model, metric: str = None) -> float:
    metric = "r2" if metric is None else metric
    X_train, X_test, y_train, y_test = _train_test_split(X, y)
    model.fit(X_train, y_train.flatten())
    y_pred = model.predict(X_test)
    metric = _cal_metric(y_test.flatten(), y_pred.flatten(), metric)
    return metric


# 紧凑关联识别
def recog_compact(X, y, features_rank, model, n_features_lst, S_pre: set=None, metric: str="r2"):
    S_pre = set() if S_pre is None else S_pre
    
    # FIMT测试
    fimt_values = []
    for n_features in n_features_lst:
        S = list(S_pre) + features_rank[:n_features]
        fimt_values.append(eval_prediction_effect(X[:, S], y, model, metric=metric))
    
    # 截断点判断
    max_value = np.max(fimt_values)
    for i, n_features in enumerate(n_features_lst):
        if fimt_values[i] > max_value - 0.05:
            best_n_features = n_features
            break
    
    # 新识别出的紧凑变量
    features_compact = features_rank[:best_n_features]
    return features_compact, fimt_values
    
    

if __name__ == "__main__":
    from sklearn.ensemble import RandomForestRegressor
    import pandas as pd
    import sys
    import os
    
    from cmim import CondMIMaximization
    
    BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
    sys.path.insert(0, BASE_DIR)
    
    from setting import plt
    
    X_df = pd.read_csv(f"{BASE_DIR}/dataset/fcc/X.csv")
    Y_df = pd.read_csv(f"{BASE_DIR}/dataset/fcc/Y.csv")
    
    # 候选特征数据和目标
    # X, X_pre = X_df.values[:, 20:], X_df.values[:, :20]
    X, X_pre = X_df.values[:, :], None
    
    # 目标数据
    y = Y_df.values[:, 3]
    
    # ---- 紧凑关联识别 -----------------------------------------------------------------------------

    X = X if X_pre is None else np.c_[X_pre, X]  # 将已选和候选拼合为一个数组
    
    S_pre = set() if X_pre is None else set(range(X_pre.shape[1]))

    self = CondMIMaximization(X, y)
    features_rank, _ = self.feature_selection(S_pre, verbose=True)
    
    # ---- 紧凑关联识别 -----------------------------------------------------------------------------
    
    metric="r2"
    model = RandomForestRegressor(
        n_estimators=100, 
        max_features="sqrt", 
        min_samples_leaf=3,
        min_samples_split=10,
        n_jobs = 3)
    n_features_lst = np.arange(1, len(features_rank) + 1, 1)  # 依次递增的测试特征数
    
    features_compact, fimt_values = recog_compact(X, y, features_rank, model, n_features_lst, S_pre, metric)
    
    plt.plot(n_features_lst, fimt_values)