"""
CE 预测器 (Cardiac Event Predictor) 模型
"""

import torch
import torch.nn as nn
from typing import Optional

# 从同级目录的 conformer.py 中导入 ConformerBlock
from .conformer import ConformerBlock

class ConvBlock(nn.Module):
    """
    常规卷积块 (不进行子采样)
    用于CE预测器，保持序列长度不变
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        # stride=1 确保序列长度不变
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.max_pool(x)
        return x

class CEPredictor(nn.Module):
    """心脏事件预测器模型"""
    
    def __init__(self, config: dict):
        super().__init__()
        
        # --- 使用不带前缀的、最直接的参数名 ---
        # 这一步将确保它能正确地从 CEPredictorConfig 和 MMwaveTrainConfig 中读取参数

        self.input_dim = getattr(config, 'ce_input_dim', config.input_dim)
        self.output_dim = getattr(config, 'ce_output_dim', config.output_dim)
        self.d_model = getattr(config, 'ce_d_model', config.d_model)
        self.num_layers = getattr(config, 'ce_num_layers', config.num_layers)
        self.num_heads = getattr(config, 'ce_num_heads', config.num_heads)
        
        conv_channels = getattr(config, 'ce_conv_channels', config.conv_channels)
        
        # 卷积块 (不进行子采样)
        
        self.conv_block1 = ConvBlock(self.input_dim, conv_channels[0])
        self.conv_block2 = ConvBlock(conv_channels[0], conv_channels[1])
        
        # 从卷积输出到d_model的线性投影
        self.linear_projection = nn.Linear(conv_channels[1], self.d_model)
        self.projection_dropout = nn.Dropout(config.projection_dropout)
        
        # Conformer块
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                expansion_factor=config.expansion_factor,
                conv_kernel_size=config.conv_kernel_size,
                dropout=config.dropout
            ) for _ in range(self.num_layers)
        ])
        
        # 最终输出层
        self.output_projection = nn.Linear(self.d_model, self.output_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: 输入mmWave信号 (batch_size, seq_len, input_dim) -> (32, 1000, 1)
            mask: 注意力掩码 (此处未使用)
            
        Returns:
            预测的心脏事件信息 (batch_size, output_dim, seq_len) -> (32, 5, 1000)
        """
        # 转换维度用于卷积处理: (N, L, C) -> (N, C, L)
        x = x.transpose(1, 2)
        
        # 通过卷积块，序列长度保持不变
        x = self.conv_block1(x) # -> (N, 16, 1000)
        x = self.conv_block2(x) # -> (N, 16, 1000)
        
        # 转换回序列格式: (N, C, L) -> (N, L, C)
        x = x.transpose(1, 2)
        
        # 线性投影到d_model
        x = self.linear_projection(x) # -> (N, 1000, 16)
        x = self.projection_dropout(x)
        
        # 通过Conformer块
        for block in self.conformer_blocks:
            x = block(x, mask)
            
        # 最终投影到输出维度
        x = self.output_projection(x) # -> (N, 1000, 5)
        
        # 转换到最终输出格式: (N, L, C_out) -> (N, C_out, L)
        x = x.transpose(1, 2) # -> (N, 5, 1000)
        
        return x

def build_ce_predictor(config: dict) -> CEPredictor:
    """构建CE预测器模型"""
    return CEPredictor(config)