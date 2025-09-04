"""
mmWave 编码器模型
"""
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

# 复用 ECG Encoder 的卷积子采样块和 Conformer 块
from .ecg_autoencoder import ConvolutionalSubsamplingBlock
from .conformer import ConformerBlock, PositionalEncoding

class mmWaveEncoder(nn.Module):
    """毫米波信号编码器"""
    
    def __init__(self, config: dict):
        super().__init__()
        
        
        # --- 使用 config.attribute 的方式访问参数 ---
        self.input_dim = config.input_dim # 应该是 6
        self.d_model = config.d_model
        
        # 卷积子采样块 (输入通道从 input_dim 开始)
        conv_channels = config.conv_channels
        
        self.conv_block1 = ConvolutionalSubsamplingBlock(self.input_dim, conv_channels[0])
        self.conv_block2 = ConvolutionalSubsamplingBlock(conv_channels[0], conv_channels[1])
        self.conv_block3 = ConvolutionalSubsamplingBlock(conv_channels[1], conv_channels[2])
        
        # 线性投影和dropout
        self.linear_projection = nn.Linear(conv_channels[2], self.d_model)
        self.projection_dropout = nn.Dropout(config.projection_dropout)
        
        # 位置编码
        # self.pos_encoding = PositionalEncoding(self.d_model, config['max_len'])
        # !! 关键修正：将 config['max_len'] 改为 config.max_len !!
        self.pos_encoding = PositionalEncoding(self.d_model, config.max_len)
        
        # Conformer块
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(
                d_model=self.d_model,
                num_heads=config.num_heads,
                expansion_factor=config.expansion_factor,
                conv_kernel_size=config.conv_kernel_size,
                dropout=config.dropout
            ) for _ in range(config.num_layers)
        ])
        
        # 最终层归一化
        self.norm = nn.LayerNorm(self.d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: 输入组合特征 (batch_size, seq_len, input_dim=6)
        Returns:
            编码后的mmWave嵌入 (batch_size, new_seq_len, d_model)
        """
        x = x.transpose(1, 2)
        
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        
        x = x.transpose(1, 2)
        
        x = self.linear_projection(x)
        x = self.projection_dropout(x)
        
        x = x * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        for block in self.conformer_blocks:
            x = block(x, mask)
            
        x = self.norm(x)
        
        return x

def build_mmwave_encoder(config: dict) -> mmWaveEncoder:
    """构建mmWave编码器"""
    return mmWaveEncoder(config)