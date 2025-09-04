# main.py

import argparse

# 导入各个阶段的配置文件
from configs.ecg_config import get_ecg_pretrain_config
from configs.ce_predictor_config import get_ce_predictor_config  # 后续阶段需要时取消注释
# from configs.mmwave_config import get_mmwave_config            # 后续阶段需要时取消注释

# 导入各个阶段的训练执行函数
from training.ecg_trainer import run_ecg_pretraining
from training.ce_trainer import run_ce_pretraining            # 后续阶段需要时取消注释
# from training.mmwave_trainer import run_mmwave_training        # 后续阶段需要时取消注释

def main():
    # --- 设置总的命令行解析器 ---
    parser = argparse.ArgumentParser(description="Radar2ECGNet 主训练脚本")
    
    # 关键参数：--stage，用来选择要执行哪个训练阶段
    parser.add_argument(
        '--stage', 
        type=str, 
        required=True, 
        choices=['pretrain_ecg', 'pretrain_ce', 'train_mmwave'],
        help="选择要执行的训练阶段"
    )
    
    # --- 在这里补充缺失的参数定义 ---
    parser.add_argument('--data_root', type=str, required=True, help='数据集的根目录路径')
    parser.add_argument('--log_dir', type=str, default='runs', help='TensorBoard日志保存目录')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='模型权重保存目录')
    parser.add_argument('--vis_interval', type=int, default=5, help='每隔多少个epoch可视化一次')
    
    args = parser.parse_args()
    
    # --- 根据stage参数，加载对应的配置并调用对应的训练函数 ---
    if args.stage == 'pretrain_ecg':
        print("--- [阶段1] 开始 ECG 自编码器预训练 ---")
        config = get_ecg_pretrain_config()
        run_ecg_pretraining(args, config)
        
    elif args.stage == 'pretrain_ce':
        print("--- [阶段2] 开始 CE 预测器预训练 ---")
        config = get_ce_predictor_config()
        run_ce_pretraining(args, config)
        # (这部分逻辑后续补充)
        
    elif args.stage == 'train_mmwave':
        print("--- [阶段3] 开始最终的 mmWave 编码器联动训练 ---")
        # config = get_mmwave_config()
        # run_mmwave_training(args, config)
        # (这部分逻辑后续补充)

if __name__ == '__main__':
    main()