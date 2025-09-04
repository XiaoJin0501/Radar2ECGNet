# configs/ecg_config.py

from .base_config import BaseConfig

class ECGPretrainConfig(BaseConfig):
    def __init__(self):
        # 首先，调用父类的 __init__ 方法来获取所有基础配置
        super().__init__()
        
        # --- 数据集默认配置 ---
        self.dataset_type = 'scenario'
        self.scenario_name = 'Resting'
        self.experiment_name = f"ecg_pretrain_{self.scenario_name.lower()}"
        
        # 实验名称（用于保存模型）
        self.experiment_name = f"ecg_pretrain_{self.scenario_name.lower()}"
        
        # 训练参数
        self.learning_rate = 1e-4
        self.epochs = 100
        
        # 模型架构 (ECGAutoencoder)
        self.model_name = "ECGAutoencoder"
        self.input_dim = 1               # ECG信号是单通道
        self.output_dim = 1
        self.d_model = 64                # 模型的核心特征维度
        self.encoder_layers = 2          # 编码器中的Conformer块数量
        self.decoder_layers = 2          # 解码器中的Conformer块数量
        self.num_heads = 64              # 多头自注意力机制的头数
        self.expansion_factor = 4        # Conformer中前馈网络的扩展因子
        self.conv_kernel_size = 31       # Conformer中卷积模块的核大小
        self.dropout = 0.1               # 模型中大部分地方的dropout率
        self.projection_dropout = 0.5    # 卷积和Conformer之间的Dropout率
        self.max_len = 5000              # 位置编码的最大序列长度
        # 卷积子采样模块的输出通道数
        self.conv_channels = [16, 32, 32] 

        # 损失权重 (根据我们之前的调试结果)
        self.pretrain_waveform_weight = 20.0
        self.pretrain_spectrogram_weight = 1.0

        # STFT参数
        self.stft_n_fft = 256
        self.stft_hop_length = 128  # 通常是 n_fft // 2
        self.stft_win_length = 256   # 通常等于 n_fft
        self.stft_window = 'hann'    # 窗函数类型
        
        # 在这里添加动态更新方法
    def update_scenario(self, scenario_name: str):
        """更新训练场景并同步实验名称"""
        self.scenario_name = scenario_name
        self.experiment_name = f"ecg_pretrain_{scenario_name.lower()}"

# 方便外部调用的函数
def get_ecg_pretrain_config():
    return ECGPretrainConfig()