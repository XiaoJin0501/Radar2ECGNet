"""
可视化工具模块 (Visualization Utilities)

提供一系列函数，用于生成模型训练过程中的对比图。
所有函数都返回一个 Matplotlib Figure 对象，以便能被 TensorBoard 记录。
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from typing import Optional
import os

def plot_ecg_reconstruction(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epoch: int,
    num_samples: int = 5,
    save_dir: str = "visualization_results"
) -> plt.Figure:
    """
    绘制真实ECG波形与重建ECG波形的对比图。
    适用于 ECG Autoencoder 预训练阶段。
    
    Args:
        y_true: 真实的ECG信号, shape (N, 1, 1000)
        y_pred: 模型重建的ECG信号, shape (N, 1, 1000)
        epoch: 当前的 Epoch 数，用于标题.
        num_samples: 要绘制的样本数量.
        save_dir: 图像保存目录
        
    Returns:
        一个 Matplotlib Figure 对象。
    """
    actual_samples = min(num_samples, y_true.shape[0])
    fig, axes = plt.subplots(actual_samples, 1, figsize=(15, 3 * actual_samples), squeeze=False)
    fig.suptitle(f'ECG Reconstruction - Epoch {epoch}', fontsize=16)

    for i in range(actual_samples):
        ax = axes[i, 0]
        ax.plot(y_true[i, 0, :], label='Ground Truth ECG', color='blue', linewidth=1.5)
        ax.plot(y_pred[i, 0, :], label='Reconstructed ECG', color='red', linestyle='--')
        ax.set_title(f'Sample {i+1}')
        ax.legend()
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Amplitude')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 保存图像
    # os.makedirs(save_dir, exist_ok=True)
    # save_path = os.path.join(save_dir, f'ecg_reconstruction_epoch_{epoch}.png')
    # plt.savefig(save_path, dpi=300)
    
    return fig

def plot_ce_prediction(
    ecg_ref: np.ndarray,         # 第一个参数应该是ECG参考信号
    ce_true: np.ndarray,         # 第二个参数是CE的真实标签
    ce_pred: np.ndarray,         # 第三个参数是CE的预测结果
    epoch: int,
    num_samples: int = 4,
    save_dir: str = "visualization_results"
) -> plt.Figure:
    """
    可视化CE预测器的预测结果。
    """
    actual_samples = min(num_samples, ecg_ref.shape[0])
    fig, axes = plt.subplots(actual_samples, 1, figsize=(15, 4 * actual_samples), squeeze=False)
    fig.suptitle(f'CE Prediction - Epoch {epoch}', fontsize=16)

    event_names = ['P', 'Q', 'R', 'S', 'T']
    event_colors = ['green', 'orange', 'red', 'purple', 'brown']

    for i in range(actual_samples):
        ax = axes[i, 0]
        ax.plot(ecg_ref[i, 0, :], color='gray', alpha=0.4, label='Ground Truth ECG (Reference)')
        
        for j in range(5):
            true_peaks = np.where(ce_true[i, j, :] == 1)[0]
            if len(true_peaks) > 0:
                ax.vlines(true_peaks, ymin=0.5, ymax=1.0, color=event_colors[j], linestyle='-', label=f'True {event_names[j]}')
            ax.plot(ce_pred[i, j, :], color=event_colors[j], linestyle='--', label=f'Predicted {event_names[j]}')

        ax.set_title(f'Sample {i+1}')
        ax.legend(loc='upper right', fontsize='small')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Amplitude / Probability')
        ax.set_ylim(-1.1, 1.1)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 保存图像
    # os.makedirs(save_dir, exist_ok=True)
    # save_path = os.path.join(save_dir, f'ce_prediction_epoch_{epoch}.png')
    # plt.savefig(save_path, dpi=300)
    
    return fig

def plot_mmwave_to_ecg_translation(
    mmwave_input: np.ndarray,
    ecg_true: np.ndarray,
    ecg_pred: np.ndarray,
    epoch: int,
    num_samples: int = 3,
    save_dir: str = "visualization_results"
) -> plt.Figure:
    """
    可视化最终mmWave到ECG的转换结果。
    """
    actual_samples = min(num_samples, mmwave_input.shape[0])
    fig, axes = plt.subplots(actual_samples, 2, figsize=(16, 3 * actual_samples), squeeze=False)
    fig.suptitle(f'mmWave-to-ECG Translation - Epoch {epoch}', fontsize=16)

    for i in range(actual_samples):
        ax_left = axes[i, 0]
        ax_left.plot(mmwave_input[i, 0, :], color='green')
        ax_left.set_title(f'Sample {i+1} - Input mmWave Signal')
        ax_left.set_xlabel('Time Steps')
        ax_left.set_ylabel('Amplitude')

        ax_right = axes[i, 1]
        ax_right.plot(ecg_true[i, 0, :], label='Ground Truth ECG', color='blue', linewidth=1.5)
        ax_right.plot(ecg_pred[i, 0, :], label='Synthesized ECG', color='red', linestyle='--')
        ax_right.set_title(f'Sample {i+1} - ECG Output')
        ax_right.legend()
        ax_right.set_xlabel('Time Steps')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 保存图像
    # os.makedirs(save_dir, exist_ok=True)
    # save_path = os.path.join(save_dir, f'mmwave_translation_epoch_{epoch}.png')
    # plt.savefig(save_path, dpi=300)
    
    
    # 添加输入验证：
    if y_true.shape != y_pred.shape:
        raise ValueError(f"形状不匹配: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    return fig