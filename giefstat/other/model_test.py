from sklearn.metrics import mean_squared_error as cal_mse
from sklearn.metrics import mean_absolute_percentage_error as cal_mape
from sklearn.metrics import r2_score as cal_r2
from sklearn.metrics import mean_absolute_error as cal_mae
import numpy as np


def exec_model_test(X, y, model, metric="r2", test_ratio=0.3, rounds=50):
    X, y = X.copy(), y.copy()
    N = X.shape[0]
    test_size = int(N * test_ratio)
    
    metrics = []
    for _ in range(rounds):
        shuffled_indexes = np.random.permutation(range(N))
        train_idxs = shuffled_indexes[test_size:]
        test_idxs = shuffled_indexes[:test_size]

        X_train, X_test = X[train_idxs, :], X[test_idxs, :]
        y_train, y_test = y[train_idxs], y[test_idxs]

        model.fit(X_train, y_train)
        
        if metric == "r2":
            metric = cal_r2(y_test, model.predict(X_test))
        elif metric == "mse":
            metric = cal_mse(y_test, model.predict(X_test))
        elif metric == "mae":
            metric = cal_mae(y_test, model.predict(X_test))
        else:
            raise ValueError(f"Invalid metric: {metric}")
        
        metrics.append(metric)
        
    return metrics