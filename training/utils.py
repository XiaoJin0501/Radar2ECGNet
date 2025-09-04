# training/utils.py

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from typing import Dict
import os # <--- 在这里添加这一行

def plot_waveform_comparison(
    model: torch.nn.Module, 
    dataloader: DataLoader, 
    epoch: int, 
    device: torch.device, 
    num_samples: int = 5
) -> plt.Figure:
    """
    创建一个包含真实波形和模型重建波形对比的 Matplotlib Figure 对象。
    
    Args:
        model: 要评估的模型.
        dataloader: 提供验证数据的 DataLoader.
        epoch: 当前的 Epoch 数，用于标题.
        device: 'cuda' 或 'cpu'.
        num_samples: 要绘制的样本数量.
        
    Returns:
        一个 Matplotlib Figure 对象，可供 TensorBoard 记录.
    """
    model.eval()  # 切换到评估模式
    
    # 从数据加载器中获取一个批次的数据
    try:
        batch = next(iter(dataloader))
        ecg_true = batch['ecg'].to(device)
    except StopIteration:
        print("Warning: Dataloader is empty, cannot generate visualization.")
        return plt.figure() # 返回一个空图像

    with torch.no_grad():
        # 假设模型输入是ecg，对于雷达项目，可能是 model(batch['radar'].to(device))
        # ↓↓↓ 在这里增加和训练/验证函数中一样的维度转换 ↓↓↓
        ecg_true_model_input = ecg_true.transpose(1, 2)
        
        # 将正确形状的数据送入模型
        ecg_pred, _ = model(ecg_true_model_input)
    
    # 将Tensor转到CPU并转换为numpy数组
    ecg_true_np = ecg_true.cpu().numpy()
    ecg_pred_np = ecg_pred.cpu().numpy()
    
    # 确保我们不会尝试绘制比批次中更多的样本
    actual_samples = min(num_samples, ecg_true.size(0))
    
    # 创建一个大图，包含 num_samples 个子图
    fig, axes = plt.subplots(actual_samples, 1, figsize=(15, 3 * actual_samples), squeeze=False)
    fig.suptitle(f'ECG Reconstruction - Epoch {epoch}', fontsize=16)

    for i in range(actual_samples):
        ax = axes[i, 0]
        ax.plot(ecg_true_np[i, 0, :], label='Ground Truth ECG', color='blue', linewidth=1.5)
        ax.plot(ecg_pred_np[i, 0, :], label='Reconstructed ECG', color='red', linestyle='--')
        ax.set_title(f'Sample {i+1}')
        ax.legend()
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Amplitude')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    model.train() # 切换回训练模式
    
    return fig

class LossManager:
    """损失管理器 - 用于跟踪和记录损失"""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """重置损失记录"""
        self.losses = {}
        self.counts = {}
        
    def update(self, loss_dict: dict, batch_size: int = 1):
        """更新损失记录"""
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
                
            if key not in self.losses:
                self.losses[key] = 0.0
                self.counts[key] = 0
                
            self.losses[key] += value * batch_size
            self.counts[key] += batch_size
            
    def get_average_losses(self) -> dict:
        """获取平均损失"""
        avg_losses = {}
        for key in self.losses:
            if self.counts[key] > 0:
                avg_losses[key] = self.losses[key] / self.counts[key]
            else:
                avg_losses[key] = 0.0
        return avg_losses
    
    def get_total_loss(self) -> float:
        """获取总损失的平均值"""
        avg_losses = self.get_average_losses()
        return avg_losses.get('total', avg_losses.get('total_loss', 0.0))
    
    #创建一个通用的保存检查点的函数，断点续训 (Resuming from a checkpoint)
def save_checkpoint(state: dict, is_best: bool, checkpoint_dir: str, experiment_name: str):
    """
    保存训练检查点。
    
    Args:
        state: 包含模型、优化器状态和epoch等信息的字典
               应该包含以下键: 'epoch', 'model_state_dict', 'optimizer_state_dict', 'best_val_loss'
        is_best: 当前模型是否是验证集上最优的
        checkpoint_dir: 检查点保存目录
        experiment_name: 实验名称，用于构成文件名
    """
    # 确保保存目录存在
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 保存最新的检查点，用于断点续训
    latest_path = os.path.join(checkpoint_dir, f"latest_{experiment_name}.pth")
    torch.save(state, latest_path)
    print(f"💾 已保存最新检查点: {latest_path}")
    
    # 如果是最佳模型，额外保存最佳模型（只保存模型权重，用于推理）
    if is_best:
        best_path = os.path.join(checkpoint_dir, f"best_{experiment_name}.pth")
        torch.save(state['model'], best_path)
        print(f"✅ 发现更优模型 (Val Loss: {state['best_val_loss']:.4f})，已保存至: {best_path}")


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None):
    """
    从检查点加载模型和优化器状态。
    
    Args:
        checkpoint_path: 检查点文件路径
        model: 要加载状态的模型
        optimizer: 要加载状态的优化器（可选）
        
    Returns:
        tuple: (start_epoch, best_val_loss) 如果成功加载，否则 (0, float('inf'))
    """
    if not os.path.exists(checkpoint_path):
        print(f"❌ 检查点文件不存在: {checkpoint_path}")
        return 0, float('inf')
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 加载模型状态
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ 已加载模型状态从: {checkpoint_path}")
        
        # 加载优化器状态（如果提供了优化器）
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"✅ 已加载优化器状态")
        
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"📊 将从 epoch {start_epoch} 继续训练，当前最佳验证损失: {best_val_loss:.4f}")
        
        return start_epoch, best_val_loss
        
    except Exception as e:
        print(f"❌ 加载检查点失败: {e}")
        return 0, float('inf')

def plot_ce_prediction(
    self,
    model: torch.nn.Module, 
    dataloader: DataLoader, 
    epoch: int, 
    device: torch.device, 
    num_samples: int = 4
) -> plt.Figure:
    """
    可视化CE预测器的预测结果。
    它会将真实ECG、真实标签和预测标签绘制在一起。
    """
    import numpy as np

    model.eval()
    
    batch = next(iter(dataloader))
    mmwave_input = batch['radar'].to(device).transpose(1, 2)
    ecg_true = batch['ecg'].cpu().numpy()
    ce_labels_true = batch['ce_labels'].cpu().numpy()

    with torch.no_grad():
        ce_labels_pred = model(mmwave_input).cpu().numpy()
    
    actual_samples = min(num_samples, mmwave_input.size(0))
    fig, axes = plt.subplots(actual_samples, 1, figsize=(15, 3 * actual_samples), squeeze=False)
    fig.suptitle(f'CE Prediction - Epoch {epoch}', fontsize=16)

    event_names = ['P', 'Q', 'R', 'S', 'T']
    event_colors = ['green', 'orange', 'red', 'purple', 'brown']

    for i in range(actual_samples):
        ax = axes[i, 0]
        # 1. 绘制真实的ECG波形作为背景参考
        ax.plot(ecg_true[i, 0, :], color='gray', alpha=0.5, label='Ground Truth ECG')
        
        # 2. 绘制真实标签和预测标签
        for j in range(5): # 遍历P,Q,R,S,T
            # 找到真实标签的位置（值为1的地方）
            true_peaks = np.where(ce_labels_true[i, j, :] == 1)[0]
            if len(true_peaks) > 0:
                ax.vlines(true_peaks, ymin=0.8, ymax=1.0, color=event_colors[j], linestyle='-', label=f'True {event_names[j]}')

            # 绘制预测值（作为一个连续的概率曲线）
            ax.plot(ce_labels_pred[i, j, :], color=event_colors[j], linestyle='--', label=f'Predicted {event_names[j]}')

        ax.set_title(f'Sample {i+1}')
        ax.legend(loc='upper right')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Amplitude / Probability')
        ax.set_ylim(-1.1, 1.1) # 设定Y轴范围以便观察

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    model.train()
    
    return fig