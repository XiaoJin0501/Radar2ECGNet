"""
training package

This package contains the core logic modules for the different 
training stages of the Radar2ECGNet project.

This __init__.py file promotes the main runner functions to the package level,
allowing for cleaner imports.

Example:
    from training import run_ecg_pretraining
    instead of
    from training.ecg_trainer import run_ecg_pretraining
"""

# 从各个子模块中导入核心的训练执行函数
from .ecg_trainer import run_ecg_pretraining
from .ce_trainer import run_ce_pretraining
from .mmwave_trainer import run_mmwave_training