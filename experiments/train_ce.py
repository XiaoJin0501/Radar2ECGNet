"""
实验脚本：第二阶段 - 预训练CE预测器

职责：
- 解析命令行参数。
- 加载CE预测器专属的配置文件。
- 调用 training/ce_trainer.py 中的核心训练逻辑。
"""

import argparse
import sys
import os

# 将项目根目录添加到Python路径中，以确保模块可以被正确导入
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入此阶段专属的配置和训练器函数
from configs.ce_predictor_config import get_ce_predictor_config
from training import run_ce_pretraining

def main():
    """CE预测器预训练的启动脚本"""
    parser = argparse.ArgumentParser(description="Stage 2: CE Predictor Pre-training")
    
    # 定义此阶段需要的命令行参数
    parser.add_argument('--data_root', type=str, required=True, help='数据集的根目录路径。')
    parser.add_argument('--log_dir', type=str, default='runs', help='TensorBoard日志保存目录。')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='模型权重保存目录。')
    parser.add_argument('--vis_interval', type=int, default=5, help='每隔多少个epoch可视化一次。')
    
    
    # --- 数据集控制参数 ---
    parser.add_argument('--dataset_type', type=str, default='scenario', choices=['scenario', 'mixed'])
    parser.add_argument('--scenario_name', type=str, default='Resting')
    
    
    args = parser.parse_args()
    
    # 1. 加载此阶段专属的配置
    config = get_ce_predictor_config()
    
    # 根据命令行参数动态更新
    config.dataset_type = args.dataset_type
    if args.dataset_type == 'scenario':
        config.scenario_name = args.scenario_name
        config.experiment_name = f"ce_pretrain_{config.scenario_name.lower()}"
    else:
        config.experiment_name = "ce_pretrain_mixed"
    
    # 3. 调用 training/ce_trainer.py 中定义的核心训练逻辑
    run_ce_pretraining(args, config)

if __name__ == '__main__':
    main()