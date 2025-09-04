# configs/mmwave_config.py

from .base_config import BaseConfig
from .ecg_config import ECGPretrainConfig
from .ce_predictor_config import CEPredictorConfig

class MMwaveTrainConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        
        # --- 数据集默认配置 ---
        self.dataset_type = 'scenario'
        self.train_scenario = 'Resting'
        
        # --- 实验配置 ---
        self.experiment_name = f"mmwave_train_{self.train_scenario.lower()}"  # 动态生成实验名

        # --- 训练参数 ---
        self.learning_rate = 1e-4
        self.epochs = 150
        
        # --- 加载其他配置作为子对象，结构更清晰 ---
        self.ecg_cfg = ECGPretrainConfig()
        self.ce_cfg = CEPredictorConfig()

        # --- mmWave Encoder 自身架构参数 ---
        self.model_name = "mmWaveEncoder"
        self.input_dim = 6  # 1 (雷达) + 5 (CE预测)
        self.d_model = self.ecg_cfg.d_model # 必须与ECG Encoder的d_model保持一致
        self.num_layers = self.ecg_cfg.encoder_layers
        self.num_heads = self.ecg_cfg.num_heads
        self.conv_channels = self.ecg_cfg.conv_channels
        # 共享通用参数
        self.expansion_factor = self.ecg_cfg.expansion_factor
        self.conv_kernel_size = self.ecg_cfg.conv_kernel_size
        self.dropout = self.ecg_cfg.dropout
        self.projection_dropout = self.ecg_cfg.projection_dropout
        self.max_len = self.ecg_cfg.max_len
        
        # --- 损失函数权重 ---
        self.embedding_loss_weight = 1.0
        self.waveform_loss_weight = 1.0
        self.spectrogram_loss_weight = 1.0
        
        # --- ↓↓↓ 在这里补充 STFT 参数 ↓↓↓ ---
        self.stft_n_fft = 256
        self.stft_hop_length = 128  # 通常是 n_fft // 2
        self.stft_win_length = 256   # 通常等于 n_fft
        self.stft_window = 'hann'    # 窗函数类型
        
        
    def update_scenario(self, train_scenario: str):
        """更新训练场景并同步实验名称"""
        self.train_scenario = train_scenario
        self.experiment_name = f"mmwave_train_{train_scenario.lower()}"
        
        # 同时更新子配置的场景（如果需要的话）
        self.ecg_cfg.update_scenario(train_scenario)
        self.ce_cfg.update_scenario(train_scenario)


# 方便外部调用的函数
def get_mmwave_config():
    return MMwaveTrainConfig()