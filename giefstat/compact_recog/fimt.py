import numpy as np

from ..util import exec_model_test


# 紧凑关联识别
def recog_compact(X, y, features_rank, model, n_features_lst, S_pre: set=None, metric: str="r2", 
                  rounds: int=3):
    # TODO: 补充说明文档        
    S_pre = set() if S_pre is None else S_pre
    
    # FIMT测试
    fimt_values = []
    for n_features in n_features_lst:
        S = list(S_pre) + features_rank[:n_features]
        fimt_values.append(exec_model_test(X[:, S], y, model, metric=metric, rounds=rounds)[0])
    
    # 截断点判断
    max_value = np.max(fimt_values)
    for i, n_features in enumerate(n_features_lst):
        if fimt_values[i] > max_value - 0.01:
            best_n_features = n_features
            break
    
    # 新识别出的紧凑变量在X所有特征中的序号
    features_compact = features_rank[:best_n_features]
    return features_compact, fimt_values
    

# if __name__ == "__main__":
#     from sklearn.ensemble import RandomForestRegressor
#     import pandas as pd
#     import sys
#     import os
    
#     from cmim import CondMIMaximization
    
#     BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
#     sys.path.insert(0, BASE_DIR)
    
#     from setting import plt
    
#     X_df = pd.read_csv(f"{BASE_DIR}/dataset/fcc/X.csv")
#     Y_df = pd.read_csv(f"{BASE_DIR}/dataset/fcc/Y.csv")
    
#     # 候选特征数据和目标
#     # X, X_pre = X_df.values[:, 20:], X_df.values[:, :20]
#     X, X_pre = X_df.values[:, :], None
    
#     # 目标数据
#     y = Y_df.values[:, 0]
    
#     # ---- 紧凑关联识别 -----------------------------------------------------------------------------

#     X = X if X_pre is None else np.c_[X_pre, X]  # 将已选和候选拼合为一个数组
    
#     S_pre = set() if X_pre is None else set(range(X_pre.shape[1]))

#     self = CondMIMaximization(X, y)
#     features_rank, _ = self.feature_selection(S_pre, verbose=True)
    
#     # ---- 紧凑关联识别 -----------------------------------------------------------------------------
    
#     metric="r2"
#     model = RandomForestRegressor(
#         n_estimators=100, 
#         max_features="sqrt", 
#         min_samples_leaf=3,
#         min_samples_split=10,
#         n_jobs = 3)
#     n_features_lst = np.arange(1, len(features_rank) + 1, 1)  # 依次递增的测试特征数
    
#     features_compact, fimt_values = recog_compact(
#         X, y, features_rank, model, n_features_lst, S_pre, metric)
    
#     plt.plot(n_features_lst, fimt_values)