"""
Conformer模块 - 用于ECG和mmWave信号的时间序列建模
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力机制"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 线性投影层
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: 输入张量 (batch_size, seq_len, d_model)
            mask: 注意力掩码 (batch_size, seq_len, seq_len) or None
        
        Returns:
            输出张量 (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.size()
        
        # 线性投影并重塑为多头格式
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # 应用掩码
        if mask is not None:
            if mask.dim() == 3:  # (batch_size, seq_len, seq_len)
                mask = mask.unsqueeze(1)  # (batch_size, 1, seq_len, seq_len)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        context = torch.matmul(attn_weights, V)
        
        # 重塑并合并多头
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # 最终线性投影
        output = self.w_o(context)
        
        return output


class ConvolutionModule(nn.Module):
    """Conformer卷积模块
    
    包含深度可分离卷积、GLU激活和批归一化
    """
    
    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        
        # 确保kernel_size为奇数，便于padding
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        self.kernel_size = kernel_size
        self.d_model = d_model
        
        # 第一个线性层：扩展维度
        self.pointwise_conv1 = nn.Linear(d_model, 2 * d_model)
        
        # GLU激活
        self.glu = nn.GLU(dim=-1)
        
        # 深度卷积
        self.depthwise_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=d_model  # 深度卷积
        )
        
        # 批归一化
        self.batch_norm = nn.BatchNorm1d(d_model)
        
        # Swish激活函数
        self.swish = nn.SiLU()
        
        # 第二个线性层：恢复维度
        self.pointwise_conv2 = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量 (batch_size, seq_len, d_model)
            
        Returns:
            输出张量 (batch_size, seq_len, d_model)
        """
        # 第一个点卷积 + GLU
        x = self.pointwise_conv1(x)  # (batch, seq_len, 2*d_model)
        x = self.glu(x)  # (batch, seq_len, d_model)
        
        # 深度卷积（需要转置维度）
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.swish(x)
        x = x.transpose(1, 2)  # (batch, seq_len, d_model)
        
        # 第二个点卷积
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        return x


class FeedForwardModule(nn.Module):
    """前馈网络模块"""
    
    def __init__(self, d_model: int, expansion_factor: int = 4, dropout: float = 0.1):
        super().__init__()
        d_ff = d_model * expansion_factor
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.swish = nn.SiLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.swish(self.linear1(x))))


class ConformerBlock(nn.Module):
    """Conformer基本块
    
    结构：FeedForward(1/2) -> MultiHeadAttention -> ConvolutionModule -> FeedForward(1/2)
    每个子模块都有残差连接和层归一化
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        expansion_factor: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 第一个前馈网络
        self.ff1 = FeedForwardModule(d_model, expansion_factor, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # 多头自注意力
        self.mhsa = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 卷积模块
        self.conv = ConvolutionModule(d_model, conv_kernel_size, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        
        # 第二个前馈网络
        self.ff2 = FeedForwardModule(d_model, expansion_factor, dropout)
        self.norm4 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: 输入张量 (batch_size, seq_len, d_model)
            mask: 注意力掩码
            
        Returns:
            输出张量 (batch_size, seq_len, d_model)
        """
        # 第一个前馈网络 (一半权重)
        residual = x
        x = self.norm1(x)
        x = self.ff1(x)
        x = self.dropout(x) * 0.5 + residual
        
        # 多头自注意力
        residual = x
        x = self.norm2(x)
        x = self.mhsa(x, mask)
        x = self.dropout(x) + residual
        
        # 卷积模块
        residual = x
        x = self.norm3(x)
        x = self.conv(x)
        x = self.dropout(x) + residual
        
        # 第二个前馈网络 (一半权重)
        residual = x
        x = self.norm4(x)
        x = self.ff2(x)
        x = self.dropout(x) * 0.5 + residual
        
        return x


class PositionalEncoding(nn.Module):
    """位置编码模块"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量 (batch_size, seq_len, d_model)
        Returns:
            添加位置编码的张量 (batch_size, seq_len, d_model)
        """
        return x + self.pe[:x.size(1), :].transpose(0, 1)


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


# 使用示例和测试
if __name__ == "__main__":
    # 测试单个ConformerBlock
    d_model = 64
    num_heads = 64
    seq_len = 1000
    batch_size = 2
    
    # 创建Conformer块
    conformer_block = ConformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        expansion_factor=4,
        conv_kernel_size=31,
        dropout=0.1
    )
    
    # 测试输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 前向传播
    output = conformer_block(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"参数数量: {sum(p.numel() for p in conformer_block.parameters())}")
    
    # 测试位置编码
    pos_encoder = PositionalEncoding(d_model=d_model)
    pos_output = pos_encoder(x)
    print(f"位置编码输出形状: {pos_output.shape}")
    
    # 测试卷积子采样块
    conv_block = ConvolutionalSubsamplingBlock(in_channels=1, out_channels=16)
    conv_input = torch.randn(batch_size, 1, seq_len)
    conv_output = conv_block(conv_input)
    print(f"卷积块输入形状: {conv_input.shape}")
    print(f"卷积块输出形状: {conv_output.shape}")