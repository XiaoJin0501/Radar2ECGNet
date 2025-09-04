#!/bin/bash

# 这是一个自动化训练脚本，用于按顺序执行Radar2ECGNet项目的所有训练阶段和场景。
# 使用 set -e 命令，如果任何一步训练失败，脚本将立即停止。
set -e

echo "=========================================================="
echo "🚀 开始 Radar2ECGNet 全流程自动化训练"
echo "=========================================================="

# --- 配置区 ---
# 只需在此处修改您的数据集根目录路径
DATA_ROOT="/home/qhh2237/Radar2ECGNet/dataset/"

# 定义要训练的独立场景列表
SCENARIOS=("Resting" "Valsalva" "Apnea")

# 检查数据目录是否存在
if [ ! -d "$DATA_ROOT" ]; then
    echo "❌ 错误：数据目录不存在: $DATA_ROOT"
    exit 1
fi

# --- 阶段一：预训练 ECG Autoencoder ---
echo ""
echo "--- [阶段 1/3] 开始预训练 ECG Autoencoder ---"

# 循环训练所有独立场景
for SCENE in "${SCENARIOS[@]}"; do
    echo ""
    echo "--> 正在训练 ECG Autoencoder (场景: $SCENE)..."
    python experiments/train_ecg.py --data_root $DATA_ROOT --dataset_type scenario --scenario_name $SCENE

    # 检查模型是否保存成功
    if [ ! -f "checkpoints/best_ecg_pretrain_${SCENE,,}.pth" ]; then
        echo "❌ Error: ECG model weight file not found, training may fail"
        exit 1
    fi
done

# 训练 mixed 数据集
echo ""
echo "--> 正在训练 ECG Autoencoder (数据集: mixed)..."
python experiments/train_ecg.py --data_root $DATA_ROOT --dataset_type mixed

echo "--- [阶段 1/3] ECG Autoencoder 预训练完成 ---"


# --- 阶段二：预训练 CE Predictor ---
echo ""
echo "--- [阶段 2/3] 开始预训练 CE Predictor ---"

# 循环训练所有独立场景
for SCENE in "${SCENARIOS[@]}"; do
    echo ""
    echo "--> 正在训练 CE Predictor (场景: $SCENE)..."
    python experiments/train_ce.py --data_root $DATA_ROOT --dataset_type scenario --scenario_name $SCENE
done

# 训练 mixed 数据集
echo ""
echo "--> 正在训练 CE Predictor (数据集: mixed)..."
python experiments/train_ce.py --data_root $DATA_ROOT --dataset_type mixed

echo "--- [阶段 2/3] CE Predictor 预训练完成 ---"


# --- 阶段三：联动训练 mmWave Encoder ---
echo ""
echo "--- [阶段 3/3] 开始联动训练 mmWave Encoder ---"

# 循环训练所有独立场景
for SCENE in "${SCENARIOS[@]}"; do
    echo ""
    echo "--> 正在训练 mmWave Encoder (场景: $SCENE)..."
    # 注意这里使用的是 --train_scenario 参数
    python experiments/train_mmwave.py --data_root $DATA_ROOT --dataset_type scenario --train_scenario $SCENE
done

# 训练 mixed 数据集
echo ""
echo "--> 正在训练 mmWave Encoder (数据集: mixed)..."
python experiments/train_mmwave.py --data_root $DATA_ROOT --dataset_type mixed

echo "--- [阶段 3/3] mmWave Encoder 联动训练完成 ---"

echo ""
echo "=========================================================="
echo "✅ 所有训练任务已成功完成！"
echo "=========================================================="