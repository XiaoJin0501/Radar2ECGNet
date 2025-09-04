# experiments/train_ecg.py

import argparse
import sys
import os

# 将项目根目录添加到Python路径中，以确保模块可以被正确导入
# 这是一种常见的做法，可以避免很多导入问题
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.ecg_config import get_ecg_pretrain_config
from training import run_ecg_pretraining

def main():
    """ECG自编码器预训练的启动脚本"""
    parser = argparse.ArgumentParser(description="Stage 1: ECG Autoencoder Pre-training")
    
    parser.add_argument('--data_root', type=str, required=True, help='数据集的根目录路径')
    parser.add_argument('--log_dir', type=str, default='runs', help='TensorBoard日志保存目录')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='模型权重保存目录')
    parser.add_argument('--vis_interval', type=int, default=5, help='每隔多少个epoch可视化一次')
    
    # --- 关键改动：增加命令行参数来控制数据集 ---
    parser.add_argument(
        '--dataset_type', 
        type=str, 
        default='scenario', # 默认为单一场景
        choices=['scenario', 'mixed'],
        help="选择数据集类型"
    )
    parser.add_argument(
        '--scenario_name', 
        type=str, 
        default='Resting', # 默认场景为 Resting
        help="当 dataset_type 为 'scenario' 时，指定具体的场景名称"
    )
    
    args = parser.parse_args()
    
    # 加载此阶段专属的配置
    config = get_ecg_pretrain_config()
    
    # 根据命令行参数动态更新配置
    if hasattr(args, 'scenario_name') and args.scenario_name:
        config.update_scenario(args.scenario_name)

    
    # 调用训练逻辑
    run_ecg_pretraining(args, config)

if __name__ == '__main__':
    main()