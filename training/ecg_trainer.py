"""
ECG自编码器预训练脚本

职责：
1. 加载配置。
2. 构建数据集、模型、损失函数和优化器。
3. 执行训练和验证循环。
4. 使用TensorBoard记录日志。
5. 保存最佳模型。
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
from tqdm import tqdm # 用于显示进度条
import logging # <--- 在这里添加这一行

# 导入您自己编写的模块和构建函数

from configs.ecg_config import get_ecg_pretrain_config
from model.ecg_autoencoder import build_ecg_autoencoder
from training.ecg_loss import build_pretrain_ecg_loss
from data.pytorch_dataset import load_mixed_dataset, load_scenario_dataset
from .utils import plot_waveform_comparison, LossManager, save_checkpoint, load_checkpoint# 假设您已将可视化函数放入utils.py
from utils.visualization import plot_ecg_reconstruction

def train_one_epoch(
    model: torch.nn.Module, 
    dataloader: DataLoader, 
    loss_fn: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    device: torch.device,
    loss_manager: LossManager
) -> None:
    """训练一个Epoch的逻辑 (增加了坏数据定位功能)"""
    model.train()
    pbar = tqdm(dataloader, desc="Training")
    
    for batch_idx, batch in enumerate(pbar):
        target_waveform = batch['ecg'].to(device)    # 形状是 (32, 1, 1000)
        target_waveform_model_input = target_waveform.transpose(1, 2) # (batch, seq_len, input_dim)  # (32, 1, 1000) -> (32, 1000, 1)
        predicted_waveform, _ = model(target_waveform_model_input) # 因为这是一个自编码器，输入和目标是相同的。将ECG真值作为输入送入模型
        total_loss, loss_dict = loss_fn(predicted=predicted_waveform, target=target_waveform) # 计算损失时，再次使用ECG真值作为对比的目标

        optimizer.zero_grad()
        total_loss.backward()
        # ↓↓↓ 在这里增加梯度裁剪 ↓↓↓
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        loss_manager.update(loss_dict, batch_size=target_waveform.size(0))
        pbar.set_postfix(loss=total_loss.item())

def validate(
    model: torch.nn.Module, 
    dataloader: DataLoader, 
    loss_fn: torch.nn.Module, 
    device: torch.device,
    loss_manager: LossManager
) -> None:
    """验证一个Epoch的逻辑"""
    model.eval()
    pbar = tqdm(dataloader, desc="Validating")
    
    with torch.no_grad():
        for batch in pbar:
            target_waveform = batch['ecg'].to(device)  # 形状是 (32, 1, 1000)
            target_waveform_model_input = target_waveform.transpose(1, 2) # (batch, seq_len, input_dim)  # (32, 1, 1000) -> (32, 1000, 1)在这里增加和训练函数中一样的维度转换
            predicted_waveform, _ = model(target_waveform_model_input) # 将正确形状的数据送入模型
            total_loss, loss_dict = loss_fn(predicted=predicted_waveform, target=target_waveform)
            loss_manager.update(loss_dict, batch_size=target_waveform.size(0))
            pbar.set_postfix(loss=total_loss.item())

def run_ecg_pretraining(args, config):
    """
    ECG自编码器预训练的主执行函数。
    """
    # 1. 环境设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = os.path.join(args.log_dir, config.experiment_name)
    writer = SummaryWriter(log_dir=log_dir)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    logging.info(f"使用设备: {device}")
    logging.info(f"TensorBoard 日志将保存在: {log_dir}")

    # 2. 数据加载 (修正了所有 config['...'] 写法)
    logging.info(f"根据配置加载数据集: 类型='{config.dataset_type}'")
    if config.dataset_type == 'mixed':
        train_loader, val_loader, _ = load_mixed_dataset(
            args.data_root, 
            batch_size=config.batch_size,
            num_workers=config.num_workers
        )
    elif config.dataset_type == 'scenario':
        scenario = config.scenario_name
        logging.info(f"正在加载单一场景: '{scenario}'")
        train_loader, val_loader, _ = load_scenario_dataset(
            args.data_root, 
            scenario=scenario, 
            batch_size=config.batch_size,
            num_workers=config.num_workers
        )
    else:
        raise ValueError(f"未知的 dataset_type: {config.dataset_type}")

    # 3. 构建模型、损失函数、优化器
    model = build_ecg_autoencoder(config).to(device)
    loss_fn = build_pretrain_ecg_loss(config)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    train_loss_manager = LossManager()
    val_loss_manager = LossManager()

# --- 检查并加载检查点 ---
    start_epoch = 0
    best_val_loss = float('inf')
    checkpoint_path = os.path.join(args.checkpoint_dir, f"latest_{config.experiment_name}.pth")
    
    if os.path.exists(checkpoint_path):
        logging.info(f"--- 发现检查点，正在从 {checkpoint_path} 恢复训练 ---")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        logging.info(f"--- 已成功恢复，将从 Epoch {start_epoch + 1} 开始 ---")


    # 4. 主训练循环
    logging.info(f"--- 开始实验: {config.experiment_name} ---")
    
    for epoch in range(start_epoch, config.epochs):                  # 修正：从 start_epoch 开始
        print(f"\n--- Epoch {epoch+1}/{config.epochs} ---")
        
        train_loss_manager.reset()
        val_loss_manager.reset()
        
        train_one_epoch(model, train_loader, loss_fn, optimizer, device, train_loss_manager)
        validate(model, val_loader, loss_fn, device, val_loss_manager)
        
        avg_train_losses = train_loss_manager.get_average_losses()
        avg_val_losses = val_loss_manager.get_average_losses()
        
        writer.add_scalars('Loss/Total', {'train': avg_train_losses['total'], 'val': avg_val_losses['total']}, epoch)
        writer.add_scalars('Loss/Waveform', {'train': avg_train_losses['waveform'], 'val': avg_val_losses['waveform']}, epoch)
        writer.add_scalars('Loss/Spectrogram', {'train': avg_train_losses['spectrogram'], 'val': avg_val_losses['spectrogram']}, epoch)
        
        print(f"Epoch {epoch+1} 结束. "
              f"Train Loss: {avg_train_losses['total']:.4f}, "
              f"Val Loss: {avg_val_losses['total']:.4f}")
        
        # 可视化调用逻辑
        if (epoch + 1) % args.vis_interval == 0:
            val_batch = next(iter(val_loader))
            ecg_true_tensor = val_batch['ecg'].to(device)
            
            # 将模型设为评估模式
            model.eval()
            with torch.no_grad():
                # 准备模型输入
                model_input = ecg_true_tensor.transpose(1, 2)
                # 获得模型输出
                ecg_pred_tensor, _ = model(model_input)
                
            # !! 关键修正：准备传递给绘图函数的参数 !!
            # 将 PyTorch Tensors 转换为 Numpy 数组
            ecg_true_numpy = ecg_true_tensor.cpu().numpy()
            ecg_pred_numpy = ecg_pred_tensor.cpu().numpy()
            
            fig = plot_ecg_reconstruction(
                y_true=ecg_true_numpy,
                y_pred=ecg_pred_numpy,
                epoch=(epoch + 1)
            )
            writer.add_figure('Validation/Reconstruction', fig, global_step=epoch)
            logging.info(f"Epoch {epoch+1}: 可视化结果已记录到 TensorBoard。")
                 
        # --- 保存检查点 ---
        current_val_loss = avg_val_losses['total']
        is_best = current_val_loss < best_val_loss
        if is_best:
            best_val_loss = current_val_loss

        checkpoint_state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'config': config
        }
        save_checkpoint(checkpoint_state, is_best, args.checkpoint_dir, config.experiment_name)
            
    writer.close()
    logging.info(f"--- 实验 {config.experiment_name} 完成 ---")
