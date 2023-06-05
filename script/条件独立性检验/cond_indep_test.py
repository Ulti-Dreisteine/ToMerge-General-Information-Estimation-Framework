import seaborn as sns
import pandas as pd
import arviz as az
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
sys.path.insert(0, BASE_DIR)

from setting import plt
from statistical_tools.surrog_indep_test import exec_surrog_indep_test


def _normalize(x: np.ndarray):
    x = x.copy()
    x_min, x_max = np.min(x), np.max(x)
    return (x - x_min) / (x_max - x_min)


def gen_data(func: str, N: int):
    """产生数据"""
    x1 = np.random.normal(0, 1, N)
    x2 = np.random.normal(0, 1, N)
    e1 = np.random.random(N) * 1e-3
    e2 = np.random.random(N) * 1e-3
    z = np.random.normal(0, 1, N)
    
    if func == "M1":
        x = x1 + z + e1
        y = x2 + z + e2
    elif func == "M2":
        x = x1 + z + e1
        y = np.power(z, 2) + e2
    elif func == "M3":
        x = x1 + z + e1
        y = 0.5 * np.sin(x1 * np.pi) + z + e2
    elif func == "M4":
        x = x1 + z + e1
        y = x1 + x2 + z + e2
    elif func == "M5":
        x = np.sqrt(np.abs(x1 * z)) + z + e1
        y = 0.25 * (x1 ** 2) * (x2 ** 2) + x2 + z + e2
    elif func == "M6":
        x = np.log(np.abs(x1 * z) + 1) + z + e1
        y = 0.5 * (x1 ** 2 * z) + x2 + z + e2
    else:
        raise ValueError(f"unknown func {func}")
    
    return _normalize(x), _normalize(y), _normalize(z)


if __name__ == "__main__":
    
    # #### 数据可视化 ###############################################################################
    
    funcs = ["M1", "M2", "M3", "M4", "M5", "M6"]
    N = 1000
    legends = [
        r"$z < z_{0.1}$",
        r"$z_{0.1} \leq z < z_{0.3}$", r"$z_{0.1} \leq z < z_{0.3}$",
        r"$z_{0.3} \leq z < z_{0.5}$", r"$z_{0.5} \leq z < z_{0.7}$",
        r"$z_{0.7} \leq z < z_{0.9}$", r"$z \geq z_{0.9}$"]
    
    _, axs = plt.subplots(3, 2, figsize=(10, 10))
    for i, func in enumerate(funcs):
        x, y, z = gen_data(func, N)
        data = pd.DataFrame(np.c_[x, y, z], columns=["x", "y", "z"])
        
        qs = np.quantile(z, (0.1, 0.3, 0.5, 0.7, 0.9))
        
        def _deter_z_layer(z, qs):
            if z < qs[0]:
                return 0
            if z >= qs[-1]:
                return 5
            for i in range(4):
                if (qs[i] <= z) & (z < qs[i + 1]):
                    return i + 1
        
        data["z_layer"] = data["z"].apply(lambda z: legends[_deter_z_layer(z, qs)])
        
        ax = axs[i // 2, i % 2]
        ax.set_title(func)
        sns.kdeplot(
            x="x", y="y", data=data, hue="z_layer", hue_order=legends, shade=True, levels=5,
            palette="rocket", alpha=0.3, gridsize=50, common_grid=False, ax=ax, zorder=-1)
        sns.scatterplot(
            x="x", y="y", data=data, hue="z_layer", hue_order=legends, ax=ax, zorder=1, alpha=0.6,
            palette="rocket", marker="o", sizes=10)
        
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
    plt.tight_layout()
    plt.savefig("数据关系可视化.png", dpi=450)
    
    # #### 条件独立检验 #############################################################################
    
    method = "CMI-GIEF"
    rounds = 300
    alpha = 0.01
    
    _, axs = plt.subplots(3, 2, figsize=(10, 10))
    for i, func in enumerate(funcs):
        x, y, z = gen_data(func, N)
        assoc, (p, indep, assocs_srg) = exec_surrog_indep_test(
            x, y, method, z=z, xtype="c", ytype="c", ztype="c", rounds=rounds, alpha=alpha)

        # 画图
        ax = axs[i // 2, i % 2]
        az.plot_posterior(
            {f"{method}_Surrog": assocs_srg}, 
            kind="hist",
            bins=50,
            ref_val=assoc,
            hdi_prob=1 - alpha * 2,
            ax=ax)
        ax.set_title(f"dataset: {func}, independence detected: {indep}", fontsize=18)
        ax.set_xlabel("CMI value")
    plt.tight_layout()
    plt.savefig("条件互信息检测值分布.png", dpi=450)
    