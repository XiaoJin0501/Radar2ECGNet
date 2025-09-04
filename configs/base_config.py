"""
基础配置文件 (Base Configuration)

包含所有训练阶段共享的参数，例如文件路径、通用训练设置等。
其他具体的配置文件将继承自这个基础配置类。
"""

import torch

class BaseConfig:
    def __init__(self):
        # --- 路径配置 (Paths) ---
        # 建议使用相对路径或环境变量，但为简单起见，先用绝对路径
        self.data_root = "/home/qhh2237/Radar2ECGNet/dataset/"
        self.log_dir = "runs"
        self.checkpoint_dir = "checkpoints"

        # --- 硬件配置 (Hardware) ---
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # --- 数据集参数 (Dataset) ---
        self.seq_len = 1000 # ECG序列长度

        # --- 通用训练参数 (Training) ---
        self.batch_size = 32
        self.num_workers = 4 # DataLoader 的工作进程数


# 方便外部调用的函数
def get_base_config():
    return BaseConfig()