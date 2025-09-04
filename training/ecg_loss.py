"""
ECG损失函数模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class WaveformLoss(nn.Module):
    """波形损失 - MSE损失"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predicted: 预测波形 (batch_size, seq_len, channels)
            target: 目标波形 (batch_size, seq_len, channels)
            
        Returns:
            波形损失值
        """
        return F.mse_loss(predicted, target)


class SpectrogramLoss(nn.Module):
    """
    频谱图损失
    新版本支持选择 MSE 或 MAE 作为损失类型。
    """
    
    def __init__(
        self,
        n_fft: int = 256,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        loss_type: str = 'mse' # 新增：损失类型参数
    ):
        super().__init__()
        
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 4
        self.win_length = win_length or n_fft
        
        # 根据 loss_type 选择损失函数
        if loss_type == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_type == 'mae':
            self.criterion = nn.L1Loss() # L1Loss 就是 MAE
        else:
            raise ValueError(f"不支持的损失类型: {loss_type}")
            
    def _compute_spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        """计算频谱图"""
        x = x.squeeze(1) 
        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length).to(x.device),
            return_complex=True
        )
        magnitude = torch.abs(stft)
        return magnitude
        
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predicted: 预测波形 (batch_size, 1, seq_len)
            target: 目标波形 (batch_size, 1, seq_len)
        """
        pred_spec = self._compute_spectrogram(predicted)
        target_spec = self._compute_spectrogram(target)
        
        # 使用在 __init__ 中定义的损失函数进行计算
        return self.criterion(pred_spec, target_spec)


class DeepFeatureLoss(nn.Module):
    """深度特征损失 - 用于比较中间层特征"""
    
    def __init__(self, loss_type: str = 'mae'):
        super().__init__()
        
        if loss_type == 'mae':
            self.criterion = nn.L1Loss()
        elif loss_type == 'mse':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
            
    def forward(
        self, 
        predicted_features: List[torch.Tensor], 
        target_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            predicted_features: 预测的中间层特征列表
            target_features: 目标中间层特征列表
            
        Returns:
            深度特征损失值
        """
        total_loss = 0.0
        num_layers = min(len(predicted_features), len(target_features))
        
        for i in range(num_layers):
            layer_loss = self.criterion(predicted_features[i], target_features[i])
            total_loss += layer_loss
            
        return total_loss / num_layers if num_layers > 0 else torch.tensor(0.0)


class ECGLoss(nn.Module):
    """ECG综合损失函数"""
    
    def __init__(self, config: dict):
        super().__init__()
        
                # --- 使用 config.attribute 的方式访问参数 ---
        self.waveform_weight = config.waveform_loss_weight
        self.spectrogram_weight = config.spectrogram_loss_weight
        self.deep_feature_weight = config.deep_feature_loss_weight
        
        # 初始化损失函数
        self.waveform_loss = WaveformLoss()
        self.spectrogram_loss = SpectrogramLoss(
            n_fft=config.stft_n_fft,
            # 假设 hop_length 和 win_length 在config中是可选的
            hop_length=getattr(config, 'stft_hop_length', None),
            win_length=getattr(config, 'stft_win_length', None),
            loss_type='mse' # 默认为mse
        )
        self.deep_feature_loss = DeepFeatureLoss(
            loss_type=config.deep_feature_loss_type
        )
        
    def forward(
        self, 
        predicted: torch.Tensor,
        target: torch.Tensor,
        predicted_embeddings: Optional[List[torch.Tensor]] = None,
        target_embeddings: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算总损失
        
        Args:
            predicted: 预测ECG波形
            target: 目标ECG波形
            predicted_embeddings: 预测的中间层特征
            target_embeddings: 目标中间层特征
            
        Returns:
            总损失和各项损失的详细信息
        """
        losses = {}
        
        # 波形损失
        waveform_loss = self.waveform_loss(predicted, target)
        losses['waveform'] = waveform_loss
        
        # 频谱图损失
        spectrogram_loss = self.spectrogram_loss(predicted, target)
        losses['spectrogram'] = spectrogram_loss
        
        # 总损失
        total_loss = (
            self.waveform_weight * waveform_loss +
            self.spectrogram_weight * spectrogram_loss
        )
        
        # 深度特征损失（如果提供了embeddings）
        if (self.deep_feature_weight > 0 and 
            predicted_embeddings is not None and 
            target_embeddings is not None):
            deep_feature_loss = self.deep_feature_loss(predicted_embeddings, target_embeddings)
            losses['deep_feature'] = deep_feature_loss
            total_loss += self.deep_feature_weight * deep_feature_loss
        
        losses['total'] = total_loss
        
        return total_loss, losses


class PretrainECGLoss(nn.Module):
    """预训练ECG自编码器损失"""
    
    def __init__(self, config: dict):
        super().__init__()
        
        # --- 使用 config.attribute 的方式访问参数 ---
        self.waveform_weight = config.pretrain_waveform_weight
        self.spectrogram_weight = config.pretrain_spectrogram_weight
        
        self.waveform_loss = WaveformLoss()
        self.spectrogram_loss = SpectrogramLoss(
            n_fft=config.stft_n_fft,
            loss_type='mse' # 预训练阶段固定使用mse
        )
        
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            predicted: 预测ECG波形
            target: 目标ECG波形
            
        Returns:
            总损失和损失详情
        """
        waveform_loss = self.waveform_loss(predicted, target)
        spectrogram_loss = self.spectrogram_loss(predicted, target)
        
        total_loss = (
            self.waveform_weight * waveform_loss +
            self.spectrogram_weight * spectrogram_loss
        )
        
        losses = {
            'waveform': waveform_loss,
            'spectrogram': spectrogram_loss,
            'total': total_loss
        }
        
        return total_loss, losses


def build_ecg_loss(config: dict) -> ECGLoss:
    """构建ECG损失函数"""
    return ECGLoss(config)


def build_pretrain_ecg_loss(config: dict) -> PretrainECGLoss:
    """构建预训练ECG损失函数"""
    return PretrainECGLoss(config)

