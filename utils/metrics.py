"""
评估指标模块 (Evaluation Metrics)

提供用于评估雷达重建ECG信号质量的函数。
主要指标包括：
- 交叉相关性 (XCorr)
- 均方误差 (MSE)
- 均方根误差 (RMSE)
"""

import numpy as np
from typing import Dict

def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    一个便捷的函数，用于计算所有核心评估指标。

    Args:
        y_true: 真实的ECG信号 (ground-truth)。
        y_pred: 模型重建的ECG信号。

    Returns:
        一个包含所有指标计算结果的字典。
    """
    metrics = {
        'xcorr': cross_correlation(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': root_mean_squared_error(y_true, y_pred)
    }
    return metrics

def cross_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算两个信号之间的交叉相关性 (Pearson correlation coefficient)。
    衡量的是整体波形的相似度。值域为 [-1, 1]，越接近1表示越相似。
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # 确保信号长度一致
    if len(y_true) != len(y_pred):
        raise ValueError("Input signals must have the same length")
        
    # 如果信号是常数（标准差为0），相关系数无定义，返回0
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
        
    # 计算皮尔逊相关系数
    corr_matrix = np.corrcoef(y_true, y_pred)
    
    return corr_matrix[0, 1]

    # 处理NaN情况
    return 0.0 if np.isnan(corr_value) else corr_value

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算两个信号之间的均方误差 (MSE)。
    衡量的是幅度偏差的平均量级。值越小越好。
    """
    return np.mean((y_true - y_pred) ** 2)

def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算两个信号之间的均方根误差 (RMSE)。
    是MSE的平方根，其单位与原始信号相同，更具解释性。值越小越好。
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))