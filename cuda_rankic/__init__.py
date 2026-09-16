"""CUDA/CUB Spearman correlation with bounded, row-wise GPU staging."""

from .api import build_info, rank_ic
from .pipeline import rank_ic_files
from .multi import rank_ic_multi_files

__version__="0.1.1"
__all__=["rank_ic", "rank_ic_files", "rank_ic_multi_files", "build_info", "__version__"]
