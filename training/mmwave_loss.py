"""
mmWave-to-ECG 联动训练的组合损失函数
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, List, Optional

# 从 ecg_loss 模块中复用基础损失组件
from .ecg_loss import WaveformLoss, SpectrogramLoss, DeepFeatureLoss

class mmWaveLoss(nn.Module):
    """mmWave联动训练综合损失函数"""
    
    def __init__(self, config: dict):
        super().__init__()
        
        # --- 使用 config.attribute 的方式访问参数 ---
        self.embedding_weight = config.embedding_loss_weight
        self.waveform_weight = config.waveform_loss_weight
        self.spectrogram_weight = config.spectrogram_loss_weight
        
        # 嵌入损失 (MAE) - 根据论文，频谱图损失也用MAE
        self.embedding_loss = DeepFeatureLoss(loss_type='mae')
        # 波形损失 (MSE)
        self.waveform_loss = WaveformLoss() 
        # 频谱图损失 (MAE) - 我们需要一个支持MAE的版本
        # 注意: SpectrogramLoss需要stft_n_fft参数，我们需要确保它在mmwave_config中定义
        self.spectrogram_loss = SpectrogramLoss(loss_type='mae', n_fft=config.stft_n_fft)

    def forward(
        self,
        predicted_waveform: torch.Tensor,
        target_waveform: torch.Tensor,
        predicted_embedding: torch.Tensor,
        target_embedding: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        
        losses = {}
        
        # 计算各项损失
        loss_embed = self.embedding_loss([predicted_embedding], [target_embedding])
        loss_wave = self.waveform_loss(predicted_waveform, target_waveform)
        loss_spec = self.spectrogram_loss(predicted_waveform, target_waveform)
        
        losses['embedding'] = loss_embed
        losses['waveform'] = loss_wave
        losses['spectrogram'] = loss_spec
        
        # 加权求和
        total_loss = (
            self.embedding_weight * loss_embed +
            self.waveform_weight * loss_wave +
            self.spectrogram_weight * loss_spec
        )
        losses['total'] = total_loss
        
        return total_loss, losses

def build_mmwave_loss(config: dict) -> mmWaveLoss:
    """构建mmWave损失函数"""
    return mmWaveLoss(config)

# 注意: 为了让SpectrogramLoss支持MAE，您可能需要稍微修改 ecg_loss.py
# 如下所示，让它可以选择损失类型：
#
# class SpectrogramLoss(nn.Module):
#     def __init__(self, loss_type: str = 'mse', ...):
#         super().__init__()
#         if loss_type == 'mse':
#             self.criterion = nn.MSELoss()
#         elif loss_type == 'mae':
#             self.criterion = nn.L1Loss()
#         ...
#
#     def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         ...
#         return self.criterion(pred_spec, target_spec)