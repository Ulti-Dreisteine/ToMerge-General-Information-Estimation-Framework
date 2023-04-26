import arviz as az
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
sys.path.insert(0, BASE_DIR)

from setting import plt
from statistical_significance.bootstrap_assoc_est import cal_bootstrap_assoc


def gen_data():
    np.random.seed(42)
    N = 100
    alpha_real = 2.5
    beta_real = 0.9
    eps = np.random.normal(0, 0.3, size=N)
    
    x = np.random.normal(10, 1, size=N)
    y_real = alpha_real + beta_real * x
    y_obs = y_real + eps
    
    return x, y_obs


if __name__ == "__main__":
    x, y = gen_data()

    plt.scatter(x, y, color="w", edgecolor="k")

    rounds = 1000
    method = "DistCorr"
    assoc_mean, assocs_bt = cal_bootstrap_assoc(x, y, method, xtype="c", ytype="c", rounds=rounds)
    az.plot_posterior({f"{method}_Bootstrap": assocs_bt}, kind="hist", bins=20, ref_val=[assoc_mean])
