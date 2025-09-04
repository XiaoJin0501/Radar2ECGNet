"""
utils package

This package contains utility modules for the Radar2ECGNet project,
such as metrics and visualization tools.
"""

# 这个文件的存在本身就将 'utils' 标记为了一个Python包。

# (可选) 为了方便调用，我们可以在这里将子模块中的核心函数提升到包的顶层。
# 这样做之后，我们就可以用 `from utils import calculate_all_metrics`
# 来代替 `from utils.metrics import calculate_all_metrics`，让调用更简洁。

from .metrics import calculate_all_metrics
from .visualization import plot_ecg_reconstruction, plot_ce_prediction, plot_mmwave_to_ecg_translation