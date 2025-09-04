"""
ECG编码器模型 - 模块化实现
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from .conformer import PositionalEncoding, ConformerBlock


class ConvolutionalSubsamplingBlock(nn.Module):
    """卷积子采样块"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.max_pool(x)
        return x


class ECGEncoder(nn.Module):
    """ECG信号编码器"""
    
    def __init__(self, config: dict):
        super().__init__()
        
        # --- 使用 config.attribute 的方式访问参数 ---
        self.input_dim = config.input_dim
        self.d_model = config.d_model
        self.num_layers = config.encoder_layers
        self.num_heads = config.num_heads
        self.expansion_factor = config.expansion_factor
        self.conv_kernel_size = config.conv_kernel_size
        self.dropout = config.dropout
        self.max_len = config.max_len
        
        conv_channels = config.conv_channels
        
        # 卷积子采样块
        self.conv_block1 = ConvolutionalSubsamplingBlock(self.input_dim, conv_channels[0])
        self.conv_block2 = ConvolutionalSubsamplingBlock(conv_channels[0], conv_channels[1])
        self.conv_block3 = ConvolutionalSubsamplingBlock(conv_channels[1], conv_channels[2])
        
        # 线性投影和dropout
        self.linear_projection = nn.Linear(conv_channels[2], self.d_model)
        self.projection_dropout = nn.Dropout(config.projection_dropout)
        
        # 位置编码
        self.pos_encoding = self._create_positional_encoding()
        
        # Conformer块 - 动态导入避免循环依赖
        self.conformer_blocks = self._create_conformer_blocks()
        
        # 最终层归一化
        self.norm = nn.LayerNorm(self.d_model)
        
    def _create_positional_encoding(self):
        """创建位置编码"""
        from .conformer import PositionalEncoding
        return PositionalEncoding(self.d_model, self.max_len)
    
    def _create_conformer_blocks(self):
        """创建Conformer块"""
        from .conformer import ConformerBlock
        return nn.ModuleList([
            ConformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                expansion_factor=self.expansion_factor,
                conv_kernel_size=self.conv_kernel_size,
                dropout=self.dropout
            ) for _ in range(self.num_layers)
        ])
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Args:
            x: 输入ECG信号 (batch_size, seq_len, input_dim)
            mask: 注意力掩码
            return_embeddings: 是否返回中间embeddings
            
        Returns:
            编码后的特征和可选的中间embeddings
        """
        # x 的形状应为 (batch, seq_len, input_dim)
        # 例如 (32, 1000, 1) 对应 batch_size=32, 序列长度=1000, 输入维度=1

        # 转换维度用于卷积处理
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)，维度变为 (batch, input_dim, seq_len) -> (batch, 1, 1000)
        
        # 卷积子采样
        x = self.conv_block1(x)  # (batch, 16, seq_len)
        x = self.conv_block2(x)  # (batch, 32, seq_len)
        x = self.conv_block3(x)  # (batch, 32, seq_len)
        
        # 转换回序列格式
        x = x.transpose(1, 2)  # (batch, seq_len, 32)
        
        # 线性投影到模型维度
        x = self.linear_projection(x)  # (batch, seq_len, d_model)
        x = self.projection_dropout(x)
        
        # 缩放和位置编码
        x = x * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # 存储中间embedding
        embeddings = []
        if return_embeddings:
            embeddings.append(x.clone())
        
        # 通过Conformer块
        for i, conformer_block in enumerate(self.conformer_blocks):
            x = conformer_block(x, mask)
            if return_embeddings:
                embeddings.append(x.clone())
        
        # 最终层归一化
        x = self.norm(x)
        
        if return_embeddings:
            return x, embeddings
        else:
            return x, None
        
# 我们可以定义一个上采样块，作为子采样块的逆操作
class UpsamplingBlock(nn.Module):
    """转置卷积上采样块"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 3):
        super().__init__()
        # ConvTranspose1d 能够增加序列长度
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=0, output_padding=0
        )
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x 的形状应为 (batch, channels, seq_len)
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x
        
class ECGDecoder(nn.Module):
    """ECG信号解码器"""
    
    def __init__(self, config: dict):
        super().__init__()
        
        # 从配置中提取参数
        self.d_model = config.d_model
        self.output_dim = config.output_dim # 输出维度通常等于输入维度
        self.num_layers = config.decoder_layers
        self.num_heads = config.num_heads
        self.expansion_factor = config.expansion_factor
        self.conv_kernel_size = config.conv_kernel_size
        self.dropout = config.dropout
        
        # 输入投影层，将 d_model 映射回卷积层期望的通道数
        self.input_projection = nn.Linear(self.d_model, 32)
        
        
        # 上采样块，与编码器的子采样过程相反
        # 37 -> 111 -> 333 -> 1000 (大致)
        # 注意：为了精确匹配长度，核大小、步长和填充需要仔细计算
        # 这里提供一个近似可行的配置
        self.upsample_block1 = nn.ConvTranspose1d(32, 32, kernel_size=3, stride=3) # 37 -> 111
        self.upsample_block2 = nn.ConvTranspose1d(32, 16, kernel_size=3, stride=3) # 111 -> 333
        self.upsample_block3 = nn.ConvTranspose1d(16, self.output_dim, kernel_size=4, stride=3) # 333 -> 1000
        
        
        # 激活函数
        self.relu = nn.ReLU()
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: 编码特征 (batch_size, seq_len=37, d_model=64)
            mask: 注意力掩码(此处未使用)
            
        Returns:
            重构的ECG信号 (batch_size, seq_len, output_dim)
        """
        
        # 投影到卷积通道数
        x = self.input_projection(x) # -> (batch, 37, 32)
        x = x.transpose(1, 2)
        
        # 通过Conformer解码块
        # for conformer_block in self.conformer_blocks:
        # x = conformer_block(x, mask)
        
        # 将维度转换为转置卷积层期望的格式 (batch, channels, seq_len)
        # x = x.transpose(1, 2) # -> (batch, 32, 37)
        
        # 逐层上采样
        x = self.relu(self.upsample_block1(x)) # -> (batch, 32, 111)
        x = self.relu(self.upsample_block2(x)) # -> (batch, 16, 333)
        x = self.upsample_block3(x) # -> (batch, 1, 1000)
        
        # 转换回序列格式
        # x = x.transpose(1, 2) # -> (batch, 1000, 1)
        
        return x


class ECGAutoencoder(nn.Module):
    """ECG自编码器"""
    
    def __init__(self, config: dict):
        super().__init__()
        
        self.config = config
        # --- 使用 getattr 来安全地访问可选参数 ---
        self.output_length = getattr(config, 'output_length', None)
        
        self.encoder = ECGEncoder(config)
        self.decoder = ECGDecoder(config)
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Args:
            x: 输入ECG信号 (batch_size, seq_len, input_dim)
            mask: 注意力掩码
            return_embeddings: 是否返回编码器的中间embeddings
            
        Returns:
            重构的ECG信号和可选的embeddings
        """
        # 编码
        encoded, embeddings = self.encoder(x, mask, return_embeddings)
        
        # 长度调整
        if self.output_length is not None and encoded.size(1) != self.output_length:
            encoded = self._adjust_sequence_length(encoded, self.output_length)
        
        # 解码
        decoded = self.decoder(encoded, mask)
        
        return decoded, embeddings
    
    def _adjust_sequence_length(self, x: torch.Tensor, target_length: int) -> torch.Tensor:
        """调整序列长度"""
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)
        x = F.interpolate(x, size=target_length, mode='linear', align_corners=False)
        x = x.transpose(1, 2)  # (batch, target_length, d_model)
        return x
    
    def encode(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """仅编码"""
        encoded, _ = self.encoder(x, mask, return_embeddings=False)
        return encoded
    
    def decode(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """仅解码"""
        return self.decoder(x, mask)


class ECGEmbeddingExtractor(nn.Module):
    """ECG特征提取器 - 用于预训练模型的特征提取"""
    
    def __init__(self, encoder: ECGEncoder, target_length: Optional[int] = None):
        super().__init__()
        
        self.encoder = encoder
        self.target_length = target_length
        
        if target_length is not None:
            self.adaptive_pool = nn.AdaptiveAvgPool1d(target_length)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """提取特征"""
        features, _ = self.encoder(x, mask, return_embeddings=False)
        
        if self.target_length is not None and features.size(1) != self.target_length:
            features = features.transpose(1, 2)
            features = self.adaptive_pool(features)
            features = features.transpose(1, 2)
        
        return features


def build_ecg_encoder(config: dict) -> ECGEncoder:
    """构建ECG编码器"""
    return ECGEncoder(config)


def build_ecg_decoder(config: dict) -> ECGDecoder:
    """构建ECG解码器"""
    return ECGDecoder(config)


def build_ecg_autoencoder(config: dict) -> ECGAutoencoder:
    """构建ECG自编码器"""
    return ECGAutoencoder(config)


def build_ecg_extractor(encoder: ECGEncoder, target_length: Optional[int] = None) -> ECGEmbeddingExtractor:
    """构建ECG特征提取器"""
    return ECGEmbeddingExtractor(encoder, target_length)