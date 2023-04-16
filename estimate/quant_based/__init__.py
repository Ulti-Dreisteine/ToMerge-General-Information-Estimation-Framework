import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 1))
sys.path.insert(0, BASE_DIR)

from mi_classic import MutualInfoClassic
from mi_darbellay import MutualInfoDarbellay

__all__ = ["MutualInfoClassic", "MutualInfoDarbellay"]