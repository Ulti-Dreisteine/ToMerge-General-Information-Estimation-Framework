from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
sys.path.insert(0, BASE_DIR)

from setting import plt
from giefstat.compact_recog import CondMIMaximization, recog_compact

# 载入数据
X_df = pd.read_csv(f"{BASE_DIR}/dataset/fcc/X.csv")
Y_df = pd.read_csv(f"{BASE_DIR}/dataset/fcc/Y.csv")

# 候选特征数据和目标
X_cand, X_pre = X_df.values[:, 20:], X_df.values[:, :10]
# X_cand, X_pre = X_df.values[:, :], None

# 目标数据
y = Y_df.values[:, 0]

# ---- 紧凑关联识别 ---------------------------------------------------------------------------------

if X_pre is None:
    X = X_cand
    S_pre = set()
else:
    X = np.c_[X_pre, X_cand]  # NOTE: 按照已选和候选先后顺序拼合为一个数组
    S_pre = set(range(X_pre.shape[1]))

self = CondMIMaximization(X, y)
features_rank, _ = self.feature_selection(S_pre, verbose=True)

# ---- 紧凑关联识别 ---------------------------------------------------------------------------------

metric="r2"
model = RandomForestRegressor(
    n_estimators=100, 
    max_features="auto", # "sqrt", 
    min_samples_leaf=3,
    min_samples_split=10,
    n_jobs = 3)
n_features_lst = np.arange(1, len(features_rank) + 1, 1)  # 依次递增的测试特征数

features_compact, fimt_values = recog_compact(
    X, y, features_rank, model, n_features_lst, S_pre, metric, rounds=3)

plt.plot(n_features_lst, fimt_values)