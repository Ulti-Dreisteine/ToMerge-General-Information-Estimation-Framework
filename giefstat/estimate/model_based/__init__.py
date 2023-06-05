import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 2))
sys.path.insert(0, BASE_DIR)

from model_based.mi_cmi import MutualInfoModel, CondMutualInfoModel

__all__ = ["MutualInfoModel", "CondMutualInfoModel"]