import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 2))
sys.path.insert(0, BASE_DIR)

from other.coefficient import cal_dist_corr, cal_pearson_corr, cal_spearman_corr

__all__ = ["cal_dist_corr", "cal_pearson_corr", "cal_spearman_corr"]