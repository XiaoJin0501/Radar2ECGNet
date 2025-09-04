"""
mmWave Encoder 联动训练脚本
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
from typing import Dict # 用于类型注解
import logging

# 导入所有需要的模型和构建函数
from model import build_ecg_autoencoder, build_ce_predictor, build_mmwave_encoder # 假设您已经在models/__init__.py中添加了这个函数
from training.mmwave_loss import build_mmwave_loss
from data.pytorch_dataset import load_scenario_dataset, load_mixed_dataset
from .utils import plot_waveform_comparison, LossManager, save_checkpoint, load_checkpoint# 假设您已将可视化函数放入utils.py
from utils.visualization import plot_mmwave_to_ecg_translation

# !! 新增：导入另外两个阶段的config !!
from configs.ecg_config import get_ecg_pretrain_config
from configs.ce_predictor_config import get_ce_predictor_config


def train_one_epoch(
    models: Dict[str, nn.Module],
    dataloader: DataLoader, 
    loss_fn: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    device: torch.device,
    loss_manager: LossManager
) -> None:
    models['mmwave_encoder'].train() # 只有mmwave_encoder需要设为训练模式
    pbar = tqdm(dataloader, desc="Training mmWave Encoder")
    
    for batch in pbar:
        radar_signal = batch['radar'].to(device)
        ecg_true = batch['ecg'].to(device)
        
        # --- 联动前向传播 ---
        with torch.no_grad():
            ce_pred = models['ce_predictor'](radar_signal.transpose(1, 2))
            target_embedding, _ = models['ecg_encoder'](ecg_true.transpose(1, 2))
            
        model_input = torch.cat([radar_signal.transpose(1, 2), ce_pred.transpose(1, 2)], dim=-1)
        predicted_embedding = models['mmwave_encoder'](model_input)
        
        with torch.no_grad():
            # !! 关键修正：ECGDecoder的输出已经是 (N, C, L) 格式 !!
            # !! 删除这行末尾的 .transpose(1, 2) !!
            predicted_waveform = models['ecg_decoder'](predicted_embedding)
            
            
            # 现在 predicted_waveform 和 ecg_true 的维度都是 (32, 1, 1000)，完全匹配

        # --- 计算总损失 ---
        total_loss, loss_dict = loss_fn(
            predicted_waveform=predicted_waveform, # <--- 直接传入 predicted_waveform
            target_waveform=ecg_true,
            predicted_embedding=predicted_embedding,
            target_embedding=target_embedding
        )
        
        # 反向传播，只会更新mmwave_encoder的参数
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(models['mmwave_encoder'].parameters(), max_norm=1.0)
        optimizer.step()
        
        loss_manager.update(loss_dict, batch_size=radar_signal.size(0))
        pbar.set_postfix(loss=total_loss.item())

def validate(
    models: Dict[str, nn.Module],
    dataloader: DataLoader, 
    loss_fn: nn.Module, 
    device: torch.device,
    loss_manager: LossManager
) -> None:
    """验证一个Epoch的逻辑"""
    # 将所有模型设为评估模式
    for model in models.values():
        model.eval()
        
    pbar = tqdm(dataloader, desc="Validating mmWave Encoder")
    
    with torch.no_grad():
        for batch in pbar:
            radar_signal = batch['radar'].to(device)
            ecg_true = batch['ecg'].to(device)
            
            # --- 完整的联动前向传播 ---
            ce_pred = models['ce_predictor'](radar_signal.transpose(1, 2))
            target_embedding, _ = models['ecg_encoder'](ecg_true.transpose(1, 2))
            model_input = torch.cat([radar_signal.transpose(1, 2), ce_pred.transpose(1, 2)], dim=-1)
            predicted_embedding = models['mmwave_encoder'](model_input)
            
            # !! 关键修正：同样删除这行末尾的 .transpose(1, 2) !!
            predicted_waveform = models['ecg_decoder'](predicted_embedding)

            # --- 计算总损失 ---
            total_loss, loss_dict = loss_fn(
                predicted_waveform=predicted_waveform, # <--- 直接传入 predicted_waveform
                target_waveform=ecg_true,
                predicted_embedding=predicted_embedding,
                target_embedding=target_embedding
            )
            
            loss_manager.update(loss_dict, batch_size=radar_signal.size(0))
            pbar.set_postfix(loss=total_loss.item())

def run_mmwave_training(args, config):
    """mmWave编码器联动训练的主执行函数"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = os.path.join(args.log_dir, config.experiment_name)
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, 'mmwave_train'))
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    logging.info(f"使用设备: {device}")
    logging.info(f"TensorBoard 日志将保存在: {log_dir}")

    # --- 1. 加载所有预训练模型 ---
    # --- 动态构建预训练模型的路径 ---
    # a. 从当前实验名中提取数据集后缀 (例如, 从 'mmwave_train_mixed' 中提取 'mixed')
    dataset_suffix = config.experiment_name.replace('mmwave_train_', '')
    # b. 构建对应的ECG和CE预训练实验名
    ecg_exp_name = f"ecg_pretrain_{dataset_suffix}"
    ce_exp_name = f"ce_pretrain_{dataset_suffix}"
    
    # c. 构建完整的权重文件路径
    ecg_checkpoint_path = os.path.join(args.checkpoint_dir, f"best_{ecg_exp_name}.pth")
    ce_checkpoint_path = os.path.join(args.checkpoint_dir, f"best_{ce_exp_name}.pth")
    
    logging.info(f"  -> 正在加载 ECG 权重: {ecg_checkpoint_path}")
    logging.info(f"  -> 正在加载 CE 权重: {ce_checkpoint_path}")

    # --- 使用动态路径加载模型 ---
    ecg_config = get_ecg_pretrain_config()
    ecg_autoencoder = build_ecg_autoencoder(ecg_config).to(device)
    ecg_autoencoder.load_state_dict(torch.load(ecg_checkpoint_path, map_location=device, weights_only=True))
    ecg_encoder = ecg_autoencoder.encoder
    ecg_decoder = ecg_autoencoder.decoder
    
    ce_config = get_ce_predictor_config()
    ce_predictor = build_ce_predictor(ce_config).to(device)
    ce_predictor.load_state_dict(torch.load(ce_checkpoint_path, map_location=device, weights_only=True))

    # 冻结所有预训练模型的参数
    for param in ecg_encoder.parameters(): param.requires_grad = False
    for param in ecg_decoder.parameters(): param.requires_grad = False
    for param in ce_predictor.parameters(): param.requires_grad = False
    
    ecg_encoder.eval()
    ecg_decoder.eval()
    ce_predictor.eval()
    logging.info("--- 预训练模型加载并冻结完毕 ---")
    
    
    # --- 2. 初始化需要训练的模型 ---
    # !! 注意：这里我们使用联动训练的config (MMwaveTrainConfig) !!
    mmwave_encoder = build_mmwave_encoder(config).to(device)
    
    models = {
        'ecg_encoder': ecg_encoder,
        'ecg_decoder': ecg_decoder,
        'ce_predictor': ce_predictor,
        'mmwave_encoder': mmwave_encoder
    }

    # 3. 准备数据、损失和优化器
    train_loader, val_loader, _ = load_scenario_dataset(args.data_root, scenario=config.train_scenario, batch_size=config.batch_size)
    loss_fn = build_mmwave_loss(config)
    optimizer = optim.Adam(mmwave_encoder.parameters(), lr=config.learning_rate)
    train_loss_manager = LossManager()
    val_loss_manager = LossManager()
    
    # 4. 检查并加载检查点（只加载 mmwave_encoder 的检查点）
    start_epoch = 0
    best_val_loss = float('inf')
    checkpoint_path = os.path.join(args.checkpoint_dir, f"latest_{config.experiment_name}.pth")
    
    if os.path.exists(checkpoint_path):
        logging.info(f"发现mmWave Encoder检查点，正在从 {checkpoint_path} 恢复训练")
        start_epoch, best_val_loss = load_checkpoint(checkpoint_path, mmwave_encoder, optimizer)
        logging.info(f"已成功恢复，将从 Epoch {start_epoch} 开始")
    else:
        logging.info("未发现mmWave Encoder检查点，将从头开始训练")

    
    # 4. 主训练循环
    logging.info("--- 开始mmWave编码器联动训练 ---")

    for epoch in range(start_epoch, config.epochs):                  # 修正：从 start_epoch 开始
        print(f"\n--- Epoch {epoch+1}/{config.epochs} ---")
        train_loss_manager.reset()
        val_loss_manager.reset()

        train_one_epoch(models, train_loader, loss_fn, optimizer, device, train_loss_manager)
        validate(models, val_loader, loss_fn, device, val_loss_manager)

        avg_train_losses = train_loss_manager.get_average_losses()
        avg_val_losses = val_loss_manager.get_average_losses()

        writer.add_scalars('Loss/Total', {'train': avg_train_losses['total'], 'val': avg_val_losses['total']}, epoch)
        
        # 记录其他子损失
        for loss_name in ['embedding', 'waveform', 'spectrogram']:
            if loss_name in avg_train_losses:
                writer.add_scalars(f'Loss/{loss_name.capitalize()}', {'train': avg_train_losses[loss_name], 'val': avg_val_losses[loss_name]}, epoch)

        print(f"Epoch {epoch+1} 结束. Train Loss: {avg_train_losses['total']:.4f}, Val Loss: {avg_val_losses['total']:.4f}")

        # 可视化
        if (epoch + 1) % args.vis_interval == 0:
            for m in models.values(): m.eval()
            val_batch = next(iter(val_loader))
            radar_vis = val_batch.radar.to(device)
            ecg_true_vis = val_batch.ecg.to(device)
            
            with torch.no_grad():
                ce_pred_vis = models.ce_predictor(radar_vis.transpose(1, 2))
                model_input_vis = torch.cat([radar_vis.transpose(1, 2), ce_pred_vis.transpose(1, 2)], dim=-1)
                embedding_vis = models.mmwave_encoder(model_input_vis)
                ecg_pred_vis = models.ecg_decoder(embedding_vis)

            fig = plot_mmwave_to_ecg_translation(
                mmwave_input=radar_vis.cpu().numpy(),
                ecg_true=ecg_true_vis.cpu().numpy(),
                ecg_pred=ecg_pred_vis.transpose(1, 2).cpu().numpy(),
                epoch=(epoch + 1),
            )
            writer.add_figure('Validation/mmWave_to_ECG_Translation', fig, global_step=epoch)
            logging.info(f"Epoch {epoch+1}: 联动训练可视化结果已记录到 TensorBoard。")
        
        # 6. 保存检查点
        current_val_loss = avg_val_losses['total']
        is_best = current_val_loss < best_val_loss
        if is_best:
            best_val_loss = current_val_loss

        # 注意：这里我们只保存 mmwave_encoder 的状态
        checkpoint_state = {
            'epoch': epoch,
            'model': models['mmwave_encoder'].state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'config': config
        }
        save_checkpoint(checkpoint_state, is_best, args.checkpoint_dir, config.experiment_name)

    writer.close()
    logging.info(f"mmWave encoder training completed: {config.experiment_name}")