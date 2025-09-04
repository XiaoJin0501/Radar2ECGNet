# experiments/train_mmwave.py

import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.mmwave_config import get_mmwave_config
from training import run_mmwave_training

def main():
    """mmWave编码器联动训练的启动脚本"""
    parser = argparse.ArgumentParser(description="Stage 3: mmWave Encoder Joint Training")
    
    parser.add_argument('--data_root', type=str, required=True, help='数据集的根目录路径')
    parser.add_argument('--log_dir', type=str, default='runs', help='TensorBoard日志保存目录')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='模型权重保存目录')
    parser.add_argument('--vis_interval', type=int, default=10, help='每隔多少个epoch可视化一次')
    
    # --- 数据集控制参数 ---
    parser.add_argument('--dataset_type', type=str, default='scenario', choices=['scenario', 'mixed'])
    parser.add_argument('--train_scenario', type=str, default='Resting')
    

    args = parser.parse_args()

    # 加载此阶段专属的配置
    config = get_mmwave_config()
    
    # 2. 使用命令行参数覆盖配置，并动态生成实验名称
    config.dataset_type = args.dataset_type
    if args.dataset_type == 'scenario':
        config.train_scenario = args.train_scenario
        config.experiment_name = f"mmwave_train_{config.train_scenario.lower()}"
    else: # mixed
        config.experiment_name = "mmwave_train_mixed"
    
    # 调用训练逻辑
    run_mmwave_training(args, config)

if __name__ == '__main__':
    main()