"""
Models Package Initialization

This file makes the model builder functions directly accessible
from the top-level 'models' package, simplifying imports in other
parts of the application.

Example:
    from models import build_ecg_autoencoder
    instead of
    from models.ecg_model import build_ecg_autoencoder
"""

# 从各个子模块中导入核心的构建函数
from .ecg_autoencoder import build_ecg_autoencoder, build_ecg_encoder, build_ecg_decoder
from .ce_predictor import build_ce_predictor
from .mmwave_encoder import build_mmwave_encoder

# (可选) 导入核心的模型类
from .ecg_autoencoder import ECGAutoencoder, ECGEncoder, ECGDecoder
from .ce_predictor import CEPredictor
from .mmwave_encoder import mmWaveEncoder