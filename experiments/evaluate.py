# experiments/evaluate.py (English Version - Detailed Multi-Stage Evaluation)

import argparse
import sys
import os
import torch
import numpy as np
from tqdm import tqdm
import logging

# Add project root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import all required modules
from model import build_ecg_autoencoder, build_ce_predictor, build_mmwave_encoder
from data.pytorch_dataset import load_scenario_dataset, load_mixed_dataset
from utils.visualization import plot_ecg_reconstruction, plot_ce_prediction, plot_mmwave_to_ecg_translation
from utils.metrics import calculate_all_metrics

# Import all configuration files
from configs.ecg_config import get_ecg_pretrain_config
from configs.ce_predictor_config import get_ce_predictor_config
from configs.mmwave_config import get_mmwave_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_ce_prediction_accuracy(ce_true: np.ndarray, ce_pred: np.ndarray) -> dict:
    """
    Calculate cardiac event (CE) prediction accuracy metrics
    Args:
        ce_true: Ground truth CE labels (N, 5, L) - 5 cardiac events
        ce_pred: Predicted CE probabilities (N, 5, L)
    Returns:
        Dictionary containing prediction accuracy for each cardiac event
    """
    event_names = ['P', 'Q', 'R', 'S', 'T']
    metrics = {}
    
    for i, event_name in enumerate(event_names):
        true_event = ce_true[:, i, :].flatten()
        pred_event = ce_pred[:, i, :].flatten()
        
        # Calculate MSE and correlation
        mse = np.mean((true_event - pred_event) ** 2)
        if np.std(true_event) > 0 and np.std(pred_event) > 0:
            corr = np.corrcoef(true_event, pred_event)[0, 1]
        else:
            corr = 0.0
            
        metrics[f'{event_name}_mse'] = mse
        metrics[f'{event_name}_corr'] = corr
    
    # Overall average
    metrics['avg_mse'] = np.mean([metrics[f'{name}_mse'] for name in event_names])
    metrics['avg_corr'] = np.mean([metrics[f'{name}_corr'] for name in event_names])
    
    return metrics

def evaluate_and_visualize(args, scenario_name: str):
    """
    Perform detailed three-stage evaluation and visualization for specified scenario
    """
    logging.info(f"\n{'='*20} Starting evaluation for scenario: {scenario_name.upper()} {'='*20}")

    # --- 1. Check model checkpoint files ---
    ecg_exp_name = f"ecg_pretrain_{scenario_name.lower()}"
    ce_exp_name = f"ce_pretrain_{scenario_name.lower()}"
    mmwave_exp_name = f"mmwave_train_{scenario_name.lower()}"
    
    ecg_checkpoint = os.path.join(args.checkpoint_dir, f"best_{ecg_exp_name}.pth")
    ce_checkpoint = os.path.join(args.checkpoint_dir, f"best_{ce_exp_name}.pth")
    mmwave_checkpoint = os.path.join(args.checkpoint_dir, f"best_{mmwave_exp_name}.pth")

    missing_files = []
    for name, path in [("ECG", ecg_checkpoint), ("CE", ce_checkpoint), ("mmWave", mmwave_checkpoint)]:
        if not os.path.exists(path):
            missing_files.append(f"{name}: {path}")
    
    if missing_files:
        logging.error(f"Error: Cannot find the following model checkpoint files:")
        for missing in missing_files:
            logging.error(f"  - {missing}")
        return

    # --- 2. Load all models ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ecg_config = get_ecg_pretrain_config()
    ce_config = get_ce_predictor_config()
    mmwave_config = get_mmwave_config()
    
    ecg_autoencoder = build_ecg_autoencoder(ecg_config).to(device)
    ecg_autoencoder.load_state_dict(torch.load(ecg_checkpoint, map_location=device, weights_only=True))
    ecg_encoder = ecg_autoencoder.encoder
    ecg_decoder = ecg_autoencoder.decoder

    ce_predictor = build_ce_predictor(ce_config).to(device)
    ce_predictor.load_state_dict(torch.load(ce_checkpoint, map_location=device, weights_only=True))
    
    mmwave_encoder = build_mmwave_encoder(mmwave_config).to(device)
    mmwave_encoder.load_state_dict(torch.load(mmwave_checkpoint, map_location=device, weights_only=True))

    ecg_autoencoder.eval()
    ce_predictor.eval()
    mmwave_encoder.eval()
    logging.info(f"--- Successfully loaded all model weights for '{scenario_name}' scenario ---")

    # --- 3. Load test dataset ---
    if scenario_name.lower() == 'mixed':
        _, _, test_loader = load_mixed_dataset(args.data_root, batch_size=args.batch_size)
    else:
        _, _, test_loader = load_scenario_dataset(args.data_root, scenario=scenario_name, batch_size=args.batch_size)
    
    # --- 4. Stage-wise evaluation ---
    ecg_reconstruction_metrics = []  # Stage 1: ECG reconstruction metrics
    ce_prediction_metrics = []       # Stage 2: CE prediction metrics
    final_translation_metrics = []   # Stage 3: Final translation metrics
    visualization_data = None
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc=f"Evaluating {scenario_name}")):
            radar_signal = batch['radar'].to(device)
            ecg_true = batch['ecg'].to(device)
            ce_labels_true = batch['ce_labels'].to(device) if 'ce_labels' in batch else None

            # === Stage 1: ECG Autoencoder Reconstruction Evaluation ===
            ecg_reconstructed, _ = ecg_autoencoder(ecg_true.transpose(1, 2))
            ecg_true_np = ecg_true.cpu().numpy()
            ecg_reconstructed_np = ecg_reconstructed.cpu().numpy()
            
            for j in range(ecg_true_np.shape[0]):
                metrics = calculate_all_metrics(ecg_true_np[j], ecg_reconstructed_np[j])
                ecg_reconstruction_metrics.append(metrics)
            
            # === Stage 2: Cardiac Event (CE) Prediction Evaluation ===
            ce_pred = ce_predictor(radar_signal.transpose(1, 2))
            if ce_labels_true is not None:
                ce_labels_true_np = ce_labels_true.cpu().numpy()
                ce_pred_np = ce_pred.cpu().numpy()
                ce_metrics = calculate_ce_prediction_accuracy(ce_labels_true_np, ce_pred_np)
                ce_prediction_metrics.append(ce_metrics)
            
            # === Stage 3: Final mmWave→ECG Translation Evaluation ===
            target_embedding, _ = ecg_encoder(ecg_true.transpose(1, 2))
            model_input = torch.cat([radar_signal.transpose(1, 2), ce_pred.transpose(1, 2)], dim=-1)
            predicted_embedding = mmwave_encoder(model_input)
            ecg_synthesized = ecg_decoder(predicted_embedding)
            
            ecg_synthesized_np = ecg_synthesized.cpu().numpy()
            for j in range(ecg_true_np.shape[0]):
                metrics = calculate_all_metrics(ecg_true_np[j], ecg_synthesized_np[j])
                final_translation_metrics.append(metrics)
            
            # Save visualization data
            if i == 0:
                visualization_data = {
                    'radar': radar_signal.cpu().numpy(),
                    'ecg_true': ecg_true_np,
                    'ecg_reconstructed': ecg_reconstructed_np,
                    'ce_labels_true': ce_labels_true_np if ce_labels_true is not None else None,
                    'ce_pred': ce_pred_np,
                    'ecg_synthesized': ecg_synthesized_np
                }

    # --- 5. Output detailed stage-wise evaluation results ---
    print(f"\n{'='*50}")
    print(f"  {scenario_name.upper()} SCENARIO - DETAILED EVALUATION RESULTS")
    print(f"{'='*50}")
    
    # 5.1 Stage 1: ECG Autoencoder Reconstruction Performance
    if ecg_reconstruction_metrics:
        avg_ecg_xcorr = np.mean([m['xcorr'] for m in ecg_reconstruction_metrics])
        avg_ecg_mse = np.mean([m['mse'] for m in ecg_reconstruction_metrics])
        avg_ecg_rmse = np.mean([m['rmse'] for m in ecg_reconstruction_metrics])
        
        print(f"\nSTAGE 1: ECG Autoencoder Reconstruction Performance")
        print(f"  - Cross Correlation (XCorr): {avg_ecg_xcorr:.4f}")
        print(f"  - Mean Squared Error (MSE):  {avg_ecg_mse:.6f}")
        print(f"  - Root Mean Squared Error (RMSE): {avg_ecg_rmse:.6f}")
    
    # 5.2 Stage 2: CE Prediction Performance
    if ce_prediction_metrics:
        avg_ce_mse = np.mean([m['avg_mse'] for m in ce_prediction_metrics])
        avg_ce_corr = np.mean([m['avg_corr'] for m in ce_prediction_metrics])
        
        print(f"\nSTAGE 2: Cardiac Event (CE) Prediction Performance")
        print(f"  - Average Correlation:       {avg_ce_corr:.4f}")
        print(f"  - Average Mean Squared Error: {avg_ce_mse:.6f}")
        
        # Detailed performance for each cardiac event
        event_names = ['P', 'Q', 'R', 'S', 'T']
        print("  - Individual Cardiac Event Performance:")
        for event in event_names:
            event_corr = np.mean([m[f'{event}_corr'] for m in ce_prediction_metrics])
            event_mse = np.mean([m[f'{event}_mse'] for m in ce_prediction_metrics])
            print(f"    {event}-wave: Correlation={event_corr:.4f}, MSE={event_mse:.6f}")
    
    # 5.3 Stage 3: Final mmWave→ECG Translation Performance
    if final_translation_metrics:
        avg_final_xcorr = np.mean([m['xcorr'] for m in final_translation_metrics])
        avg_final_mse = np.mean([m['mse'] for m in final_translation_metrics])
        avg_final_rmse = np.mean([m['rmse'] for m in final_translation_metrics])
        
        print(f"\nSTAGE 3: Final mmWave→ECG Translation Performance")
        print(f"  - Cross Correlation (XCorr): {avg_final_xcorr:.4f}")
        print(f"  - Mean Squared Error (MSE):  {avg_final_mse:.6f}")
        print(f"  - Root Mean Squared Error (RMSE): {avg_final_rmse:.6f}")
    
    # --- 6. Generate visualizations ---
    if visualization_data:
        base_save_dir = os.path.join("visualization_results", scenario_name.lower())
        logging.info(f"--- Generating multi-stage visualization images for {scenario_name}... ---")
        
        # ECG reconstruction visualization
        ecg_save_dir = os.path.join(base_save_dir, "ecg_reconstruction")
        fig1 = plot_ecg_reconstruction(
            y_true=visualization_data['ecg_true'],
            y_pred=visualization_data['ecg_reconstructed'],
            epoch=0, num_samples=5, save_dir=ecg_save_dir
        )
        
        # CE prediction visualization
        if visualization_data['ce_labels_true'] is not None:
            ce_save_dir = os.path.join(base_save_dir, "ce_prediction")
            fig2 = plot_ce_prediction(
                ecg_ref=visualization_data['ecg_true'],
                ce_true=visualization_data['ce_labels_true'],
                ce_pred=visualization_data['ce_pred'],
                epoch=0, num_samples=4, save_dir=ce_save_dir
            )
        
        # Final translation visualization
        mmwave_save_dir = os.path.join(base_save_dir, "mmwave_translation")
        fig3 = plot_mmwave_to_ecg_translation(
            mmwave_input=visualization_data['radar'],
            ecg_true=visualization_data['ecg_true'],
            ecg_pred=visualization_data['ecg_synthesized'],
            epoch=0, num_samples=3, save_dir=mmwave_save_dir
        )
        
        logging.info(f"All visualization results saved to: {base_save_dir}")
        
        import matplotlib.pyplot as plt
        plt.close('all')
    
def main():
    parser = argparse.ArgumentParser(description="Comprehensive Multi-Stage Model Evaluation Script")
    
    parser.add_argument('--data_root', type=str, required=True, help='Root directory path of the dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory containing all model checkpoints')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument(
        '--scenario', 
        type=str, 
        default='All', 
        help="Scenario to evaluate. Options: 'Resting', 'Valsalva', 'Apnea', 'mixed', or 'All' to run all scenarios."
    )

    args = parser.parse_args()

    if args.scenario.lower() == 'all':
        scenarios_to_run = ['Resting', 'Valsalva', 'Apnea', 'mixed']
    else:
        scenarios_to_run = [args.scenario]
    
    for scenario in scenarios_to_run:
        evaluate_and_visualize(args, scenario)

if __name__ == '__main__':
    main()