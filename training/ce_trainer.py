"""
CE 预测器 (Cardiac Event Predictor) 预训练脚本

职责：
1. 加载配置。
2. 构建毫米波数据集、CE预测器模型、损失函数和优化器。
3. 执行训练和验证循环。
4. 使用TensorBoard记录日志。
5. 保存最佳模型。
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import logging # <--- 在这里添加这一行

# 导入您自己编写的模块和构建函数
from configs.ce_predictor_config import get_ce_predictor_config
from model.ce_predictor import build_ce_predictor
from data.pytorch_dataset import load_scenario_dataset, load_mixed_dataset # 假设我们同样使用分场景的数据
from .utils import plot_waveform_comparison, LossManager, save_checkpoint, load_checkpoint # 假设您已将可视化函数放入utils.py
from utils.visualization import plot_ce_prediction


def train_one_epoch(
    model: torch.nn.Module, 
    dataloader: DataLoader, 
    loss_fn: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    device: torch.device,
    loss_manager: LossManager
) -> None:
    """训练一个Epoch的逻辑"""
    model.train()
    pbar = tqdm(dataloader, desc="Training CE Predictor")
    
    for batch in pbar:
        # 从数据加载器获取输入和真实标签
        mmwave_signal = batch['radar'].to(device)
        ce_labels_true = batch['ce_labels'].to(device)
        
        # 准备模型输入 (N, C, L) -> (N, L, C)
        model_input = mmwave_signal.transpose(1, 2)
        
        # 前向传播
        ce_labels_pred = model(model_input)
        
        # 计算损失
        loss = loss_fn(ce_labels_pred, ce_labels_true)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 更新并记录损失
        loss_manager.update({'total_loss': loss}, batch_size=mmwave_signal.size(0))
        pbar.set_postfix(loss=loss.item())

def validate(
    model: torch.nn.Module, 
    dataloader: DataLoader, 
    loss_fn: torch.nn.Module, 
    device: torch.device,
    loss_manager: LossManager
) -> None:
    """验证一个Epoch的逻辑"""
    model.eval()
    pbar = tqdm(dataloader, desc="Validating CE Predictor")
    
    with torch.no_grad():
        for batch in pbar:
            mmwave_signal = batch['radar'].to(device)
            ce_labels_true = batch['ce_labels'].to(device)
            
            model_input = mmwave_signal.transpose(1, 2)
            ce_labels_pred = model(model_input)
            
            loss = loss_fn(ce_labels_pred, ce_labels_true)
            
            loss_manager.update({'total_loss': loss}, batch_size=mmwave_signal.size(0))
            pbar.set_postfix(loss=loss.item())

def run_ce_pretraining(args, config):
    """
    CE 预测器预训练的主执行函数
    由 main.py 调用
    
    Args:
        args: 来自 main.py 的命令行参数对象
        config: 从 configs/ce_predictor_config.py 加载的配置字典
    """
    # 1. 设置环境
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = os.path.join(args.log_dir, config.experiment_name)
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, 'ce_predictor_pretrain'))
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    logging.info(f"使用设备: {device}")
    logging.info(f"TensorBoard 日志将保存在: {log_dir}")
    
    # 2. 构建数据加载器
    train_loader, val_loader, _ = load_scenario_dataset(
        args.data_root, 
        scenario=config.scenario_name, # 改为 config.scenario_name
        batch_size=config.batch_size   # 改为 config.batch_size
    )
    
    # 3. 构建模型和损失函数
    model = build_ce_predictor(config).to(device)
    # 根据描述，损失函数就是简单的MSE
    loss_fn = nn.MSELoss()
    
    # 4. 初始化优化器和损失管理器
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    train_loss_manager = LossManager()
    val_loss_manager = LossManager()
    
    #  检查并加载检查点
    start_epoch = 0
    best_val_loss = float('inf')
    checkpoint_path = os.path.join(args.checkpoint_dir, f"latest_{config.experiment_name}.pth")
    
    if os.path.exists(checkpoint_path):
        logging.info(f"发现检查点，正在从 {checkpoint_path} 恢复训练")
        start_epoch, best_val_loss = load_checkpoint(checkpoint_path, model, optimizer)
        logging.info(f"已成功恢复，将从 Epoch {start_epoch} 开始")
    else:
        logging.info("未发现检查点，将从头开始训练")
    
    # 5. 主训练循环
    logging.info(f"开始CE预测器预训练: {config.experiment_name}")
    
    for epoch in range(start_epoch, config.epochs):                  # 修正：从 start_epoch 开始
        print(f"\n--- Epoch {epoch+1}/{config.epochs} ---")
        
        train_loss_manager.reset()
        val_loss_manager.reset()
        
        train_one_epoch(model, train_loader, loss_fn, optimizer, device, train_loss_manager)
        validate(model, val_loader, loss_fn, device, val_loss_manager)
        
        avg_train_loss = train_loss_manager.get_total_loss()
        avg_val_loss = val_loss_manager.get_total_loss()
        
        # 记录到TensorBoard
        writer.add_scalars('Loss', {'train': avg_train_loss, 'val': avg_val_loss}, epoch)
        print(f"Epoch {epoch+1} 结束. Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # 可视化调用逻辑
        if (epoch + 1) % args.vis_interval == 0:
            model.eval()
            
            # 2. 从验证集中取一个批次用于可视化
            val_batch = next(iter(val_loader))
            radar_vis = val_batch['radar'].to(device)
            ecg_ref_vis = val_batch['ecg'].cpu().numpy()
            ce_labels_true_vis = val_batch['ce_labels'].cpu().numpy()
            
            # 3. 使用模型进行一次预测
            with torch.no_grad():
                model_input_vis = radar_vis.transpose(1, 2)
                ce_labels_pred_vis = model(model_input_vis).cpu().numpy()
            
            # 4. 调用绘图函数，并传入正确的Numpy数组
            fig = plot_ce_prediction(
                ecg_ref=ecg_ref_vis,
                ce_true=ce_labels_true_vis,
                ce_pred=ce_labels_pred_vis,
                epoch=(epoch + 1)
            )
            
            # 5. 将图像写入TensorBoard
            writer.add_figure('Validation/CE_Prediction', fig, global_step=epoch)
            logging.info(f"Epoch {epoch+1}: 可视化结果已记录到 TensorBoard。")
            
            # 记得将模型切换回训练模式
            model.train()

        # 7. 保存检查点
        current_val_loss = avg_val_loss
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
    print("--- CE预测器预训练完成 ---")