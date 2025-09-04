"""
Complete Dataset Loader for Radar2ECGNet

Streamlined module for batch processing .mat files to paper-compliant .npy segments.
Processing Pipeline: 原始.mat文件 → 信号提取 → 完整预处理 → 分段处理 → 保存论文兼容格式.npy文件

Author: Radar2ECGNet Team
Version: 3.2 (Subject-organized mixed data)
"""

import os
import numpy as np
import scipy.io as sio
from scipy.signal import find_peaks
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
import warnings
from scipy.interpolate import interp1d
from datetime import datetime

# Import preprocessing pipeline
try:
    from signal_processing import (
        process_24ghz_radar_to_ecg, 
        SignalResampling,
        AdaptiveFiltering,
        BaselineCorrection,
        SignalNormalization,
        DataCleaning
    )
except ImportError:
    print("Warning: signal_processing module not found. Please ensure it's in the same directory.")

warnings.filterwarnings('ignore')


def generate_ce_labels_from_ecg(ecg_segments: np.ndarray) -> np.ndarray:
    """
    从ECG波形分段中生成心脏事件（PQRST）标签。

    Args:
        ecg_segments: ECG波形数据，形状为 (N, 1, 1000) 的Numpy数组。

    Returns:
        一个形状为 (N, 5, 1000) 的Numpy数组，其中5个通道分别对应P,Q,R,S,T波
        在峰值位置标记为1，其余为0。
    """
    
    num_segments, _, seq_len = ecg_segments.shape
    # 初始化一个全零的标签数组
    ce_labels = np.zeros((num_segments, 5, seq_len), dtype=np.float32)

    # 将 (N, 1, 1000) 的数据展平为 (N, 1000) 以便处理
    ecg_segments_flat = ecg_segments.squeeze(axis=1)

    # 遍历每一个ECG段
    for i in range(num_segments):
        ecg = ecg_segments_flat[i]

        # --- 1. 寻找R波 ---
        # R波是信号中最显著的波峰，我们首先定位它
        # height=0.1 表示波峰至少高于0.1
        # distance=100 表示两个R峰之间至少相隔100个点（对应心率不高于300bpm）
        r_peaks, _ = find_peaks(ecg, height=0.1, distance=100)
        if len(r_peaks) > 0:
            ce_labels[i, 2, r_peaks] = 1  # 第2个通道是R波

        # --- 2. 基于R波位置寻找其他波峰 ---
        for r_idx in r_peaks:
            # 定义相对于R峰的搜索窗口（这些值是基于经验，可能需要微调）
            p_window = ecg[max(0, r_idx - 150) : r_idx - 50]
            q_window = ecg[max(0, r_idx - 50) : r_idx]
            s_window = ecg[r_idx : min(seq_len, r_idx + 50)]
            t_window = ecg[min(seq_len, r_idx + 50) : min(seq_len, r_idx + 300)]

            # --- 寻找Q波 (R峰前的最低点) ---
            if len(q_window) > 0:
                q_idx_relative = np.argmin(q_window)
                q_idx = max(0, r_idx - 50) + q_idx_relative
                ce_labels[i, 1, q_idx] = 1  # 第1个通道是Q波

            # --- 寻找S波 (R峰后的最低点) ---
            if len(s_window) > 0:
                s_idx_relative = np.argmin(s_window)
                s_idx = r_idx + s_idx_relative
                ce_labels[i, 3, s_idx] = 1  # 第3个通道是S波

            # --- 寻找P波 (Q峰前的最高点) ---
            if len(p_window) > 0:
                p_peaks_relative, _ = find_peaks(p_window, height=0.01, distance=50)
                if len(p_peaks_relative) > 0:
                    # 通常取离R峰最近的那个P峰
                    p_idx = max(0, r_idx - 150) + p_peaks_relative[-1]
                    ce_labels[i, 0, p_idx] = 1  # 第0个通道是P波

            # --- 寻找T波 (S峰后的最高点) ---
            if len(t_window) > 0:
                t_peaks_relative, _ = find_peaks(t_window, height=0.01, distance=50)
                if len(t_peaks_relative) > 0:
                    # 通常取窗口内的最大值作为T峰
                    t_idx = min(seq_len, r_idx + 50) + np.argmax(t_window)
                    ce_labels[i, 4, t_idx] = 1  # 第4个通道是T波

    return ce_labels


class DatasetLoader:
    """Paper-compliant dataset loader with ECG preprocessing."""

    def __init__(self, 
                 raw_dir: str = "/Users/XiaoJin/radar2ecg_rawdatasets/",
                 output_dir: str = "/Users/XiaoJin/Radar2ECGNet/data/",
                 liu_compatible: bool = True,
                 generate_ce_labels: bool = True,
                 paper_compliant: bool = True):

        # Core configuration
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        self.liu_compatible = liu_compatible
        self.generate_ce_labels = generate_ce_labels
        self.paper_compliant = paper_compliant

        # Dataset parameters
        self.scenes = ['Resting', 'Valsalva', 'Apnea']
        self.scene_codes = {'Resting': '1', 'Valsalva': '2', 'Apnea': '3'}
        self.subjects = [f"GDN{i:04d}" for i in range(1, 31)]  # GDN0001-GDN0030

        # Signal parameters
        self.radar_fs = 2000.0
        self.target_fs = 128.0
        self.ecg_ref = 'tfm_ecg1'

        # Processing stats
        self.stats = {'processed': 0, 'failed': 0, 'total_segments': 0}

    def load_mat_signals(self, mat_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load I, Q, and ECG signals from .mat file."""

        if not os.path.exists(mat_path):
            raise FileNotFoundError(f"File not found: {mat_path}")

        # Load .mat file
        mat_data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        data = {k: v for k, v in mat_data.items() if not k.startswith('__')}

        # Find I signal
        i_names = ['radar_i', 'I', 'i_signal', 'i_data', 'I_signal', 'radar_I']
        i_signal = None
        for name in i_names:
            if name in data:
                i_signal = np.asarray(data[name]).flatten()
                break

        # Find Q signal  
        q_names = ['radar_q', 'Q', 'q_signal', 'q_data', 'Q_signal', 'radar_Q']
        q_signal = None
        for name in q_names:
            if name in data:
                q_signal = np.asarray(data[name]).flatten()
                break

        # Find ECG signal (tfm_ecg1 preferred)
        ecg_signal = np.array([])
        if self.ecg_ref in data:
            ecg_signal = np.asarray(data[self.ecg_ref]).flatten()
        else:
            # Try fallback ECG names
            ecg_names = ['ECG', 'ecg', 'ecg_signal', 'tfm_ecg2', 'tfm_ecg']
            for name in ecg_names:
                if name in data:
                    ecg_signal = np.asarray(data[name]).flatten()
                    break

        # Validate signals
        if i_signal is None or q_signal is None:
            available_keys = list(data.keys())
            raise ValueError(f"I/Q signals not found. Available keys: {available_keys}")

        if len(i_signal) != len(q_signal):
            raise ValueError(f"I/Q length mismatch: {len(i_signal)} vs {len(q_signal)}")

        # Trim to common length if ECG exists
        if len(ecg_signal) > 0:
            min_len = min(len(i_signal), len(ecg_signal))
            i_signal, q_signal, ecg_signal = i_signal[:min_len], q_signal[:min_len], ecg_signal[:min_len]

        return i_signal, q_signal, ecg_signal

    def preprocess_ecg_signal(self, ecg_signal: np.ndarray) -> np.ndarray:
        """
        对ECG信号进行完整的预处理，使用signal_processing模块中的ECG预处理功能。
        
        Args:
            ecg_signal: 原始ECG信号
            
        Returns:
            预处理后的ECG信号
        """
        if len(ecg_signal) == 0:
            return ecg_signal
        
        try:
            # Step 1: ECG带通滤波 (0.5-40Hz)
            filtered_signal = AdaptiveFiltering.ecg_bandpass_filter(
                ecg_signal, self.radar_fs, lowcut=0.5, highcut=40.0, order=4
            )
            
            # Step 2: 50Hz陷波滤波 (去除电源线干扰)
            filtered_signal = AdaptiveFiltering.adaptive_notch_filter(
                filtered_signal, self.radar_fs, target_freq=50.0, Q=30.0
            )
            
            # Step 3: 基线漂移校正
            corrected_signal = BaselineCorrection.adaptive_baseline_correction(
                filtered_signal, method='polynomial', order=5, fs=self.radar_fs
            )
            
            # Step 4: 重采样到目标采样率
            if self.radar_fs != self.target_fs:
                resampled_signal = SignalResampling.resample_with_antialiasing(
                    corrected_signal, self.radar_fs, self.target_fs
                )
            else:
                resampled_signal = corrected_signal
            
            # Step 5: 数据清理（异常值处理）
            cleaned_signal, _ = DataCleaning.remove_outliers(
                resampled_signal, method='modified_zscore', threshold=3.5
            )
            
            # Step 6: 信号归一化
            normalized_signal = SignalNormalization.minmax_normalize(
                cleaned_signal, feature_range=(-1, 1)
            )
            
            return normalized_signal
            
        except Exception as e:
            print(f"ECG preprocessing failed: {e}")
            # 如果预处理失败，至少进行基本的重采样和归一化
            if self.radar_fs != self.target_fs:
                resampled_signal = SignalResampling.resample_with_antialiasing(
                    ecg_signal, self.radar_fs, self.target_fs
                )
            else:
                resampled_signal = ecg_signal
            
            return SignalNormalization.minmax_normalize(resampled_signal, (-1, 1))

    def process_signals(self, i_signal: np.ndarray, q_signal: np.ndarray, 
                       ecg_signal: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Process radar and ECG signals into segments."""

        # Process radar signals
        results = process_24ghz_radar_to_ecg(
            i_signal=i_signal,
            q_signal=q_signal,
            output_format='liu_compatible' if self.liu_compatible else 'standard'
        )

        if not results['success']:
            raise RuntimeError(f"Radar processing failed: {results.get('error', 'Unknown error')}")

        # Get radar segments
        if self.liu_compatible:
            radar_segments = results.get('segments_1000', results.get('segments_1x1000', results['segments']))
        else:
            radar_segments = results['segments']

        # Process ECG if available
        ecg_segments = None
        ce_labels = None

        if len(ecg_signal) > 0:
            try:
                # 使用完整的ECG预处理流程
                ecg_preprocessed = self.preprocess_ecg_signal(ecg_signal)

                # Segment ECG (2s windows, 0.5s overlap)
                window_size = int(2.0 * self.target_fs)  # 256 samples
                step_size = int(1.5 * self.target_fs)    # 192 samples (0.5s overlap)

                ecg_segments_list = []
                for i in range(0, len(ecg_preprocessed) - window_size + 1, step_size):
                    segment = ecg_preprocessed[i:i + window_size]
                    if len(segment) == window_size:
                        ecg_segments_list.append(segment)

                if ecg_segments_list:
                    # Convert to Liu format if needed
                    if self.liu_compatible:
                        ecg_1000_list = []
                        for seg in ecg_segments_list:
                            # Interpolate ECG to 1000 samples
                            x_256 = np.linspace(0, 2.0, len(seg))
                            x_1000 = np.linspace(0, 2.0, 1000)
                            interpolator = interp1d(x_256, seg, kind='cubic', fill_value='extrapolate')
                            ecg_1000_list.append(interpolator(x_1000))
                        ecg_segments = np.array(ecg_1000_list)
                    else:
                        ecg_segments = np.array(ecg_segments_list)

                    # Match segment counts
                    min_segments = min(len(radar_segments), len(ecg_segments))
                    radar_segments = radar_segments[:min_segments]
                    ecg_segments = ecg_segments[:min_segments]

            except Exception as e:
                print(f"ECG processing failed: {e}")
                ecg_segments = None

        return radar_segments, ecg_segments, ce_labels

    def apply_paper_compliant_format(self, radar_segments: np.ndarray, 
                                   ecg_segments: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply paper-compliant formatting: add channel dimensions."""

        if not self.paper_compliant:
            return radar_segments, ecg_segments

        # Add channel dimension to radar: (n, seq_len) → (n, 1, seq_len)
        radar_formatted = radar_segments[:, np.newaxis, :]

        # Add channel dimension to ECG: (n, seq_len) → (n, 1, seq_len)
        ecg_formatted = None
        if ecg_segments is not None:
            ecg_formatted = ecg_segments[:, np.newaxis, :]

        return radar_formatted, ecg_formatted

    def save_segments(self, output_path: str, radar_segments: np.ndarray, 
                     ecg_segments: Optional[np.ndarray], 
                     ce_labels: Optional[np.ndarray],
                     metadata: Dict):
        """Save processed segments and metadata."""

        os.makedirs(output_path, exist_ok=True)

        # Apply paper-compliant formatting
        radar_formatted, ecg_formatted = self.apply_paper_compliant_format(radar_segments, ecg_segments)

        # Save radar segments
        radar_path = os.path.join(output_path, 'radar_segments.npy')
        np.save(radar_path, radar_formatted)

        # Save ECG segments if available
        if ecg_formatted is not None:
            ecg_path = os.path.join(output_path, 'ecg_segments.npy')
            np.save(ecg_path, ecg_formatted)
            
            # Generate and save CE labels if requested
            if self.generate_ce_labels:
                ce_labels = generate_ce_labels_from_ecg(ecg_formatted)
                ce_path = os.path.join(output_path, 'ce_labels.npy')
                np.save(ce_path, ce_labels)

        # Update metadata
        metadata.update({
            'radar_shape': radar_formatted.shape,
            'ecg_shape': ecg_formatted.shape if ecg_formatted is not None else None,
            'ce_labels_shape': ce_labels.shape if ce_labels is not None else None,
            'paper_compliant': self.paper_compliant,
            'created_at': datetime.now().isoformat()
        })

        # Save metadata
        metadata_path = os.path.join(output_path, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

    def process_single_file(self, subject: str, scene: str) -> Optional[Dict]:
        """Process a single subject-scene combination."""

        # Build file path
        code = self.scene_codes[scene]
        filename = f"{subject}_{code}_{scene}.mat"
        mat_path = os.path.join(self.raw_dir, subject, filename)

        try:
            # Load signals
            i_signal, q_signal, ecg_signal = self.load_mat_signals(mat_path)

            # Check minimum duration (30 seconds)
            duration = len(i_signal) / self.radar_fs
            if duration < 30.0:
                raise ValueError(f"Signal too short: {duration:.1f}s")

            # Process signals
            radar_segments, ecg_segments, ce_labels = self.process_signals(i_signal, q_signal, ecg_signal)

            # Prepare metadata
            metadata = {
                'subject': subject,
                'scene': scene, 
                'file_path': mat_path,
                'signal_duration': duration,
                'num_segments': len(radar_segments),
                'has_ecg': ecg_segments is not None,
                'has_ce_labels': self.generate_ce_labels and ecg_segments is not None,
                'format': 'liu_compatible' if self.liu_compatible else 'standard',
                'paper_compliant': self.paper_compliant,
                'ecg_preprocessing_applied': True,
                'sampling_rates': {
                    'original': self.radar_fs,
                    'target': self.target_fs
                }
            }

            # Save individual file
            output_path = os.path.join(self.output_dir, scene, subject)
            self.save_segments(output_path, radar_segments, ecg_segments, ce_labels, metadata)

            # Update stats
            self.stats['processed'] += 1
            self.stats['total_segments'] += len(radar_segments)

            return metadata

        except Exception as e:
            print(f"Error processing {subject}-{scene}: {str(e)}")
            self.stats['failed'] += 1
            return None

    def process_scene(self, scene: str) -> Dict[str, Any]:
        """Process all subjects for a single scene."""

        if scene not in self.scenes:
            raise ValueError(f"Invalid scene: {scene}")

        results = []
        for subject in self.subjects:
            result = self.process_single_file(subject, scene)
            if result:
                results.append(result)

        summary = {
            'scene': scene,
            'processed_subjects': len(results),
            'total_subjects': len(self.subjects),
            'total_segments': sum(r['num_segments'] for r in results),
            'success_rate': len(results) / len(self.subjects) * 100
        }

        return summary

    def process_subject_mixed(self, subject: str) -> Dict[str, Any]:
        """
        处理单个受试者在所有场景下的数据，合并为mixed格式。
        
        Args:
            subject: 受试者ID (如 'GDN0001')
            
        Returns:
            处理结果字典，包含合并后的数据信息
        """
        
        subject_radar = []
        subject_ecg = []
        subject_ce_labels = []
        subject_metadata = []
        
        # 遍历所有场景
        for scene in self.scenes:
            try:
                # 尝试从已处理的单场景数据中加载
                scene_dir = os.path.join(self.output_dir, scene, subject)
                
                if not os.path.exists(scene_dir):
                    # 如果场景数据不存在，则重新处理
                    print(f"Processing {subject}-{scene} for mixed dataset...")
                    result = self.process_single_file(subject, scene)
                    if not result:
                        continue
                
                # 加载radar数据
                radar_path = os.path.join(scene_dir, 'radar_segments.npy')
                if os.path.exists(radar_path):
                    radar_data = np.load(radar_path)
                    subject_radar.append(radar_data)
                    
                    # 为每个segment添加场景信息
                    for i in range(len(radar_data)):
                        subject_metadata.append({
                            'subject': subject,
                            'scene': scene,
                            'segment_index': i,
                            'global_segment_index': len(subject_metadata)
                        })
                
                # 加载ECG数据
                ecg_path = os.path.join(scene_dir, 'ecg_segments.npy')
                if os.path.exists(ecg_path):
                    ecg_data = np.load(ecg_path)
                    subject_ecg.append(ecg_data)
                
                # 加载CE labels数据
                ce_path = os.path.join(scene_dir, 'ce_labels.npy')
                if os.path.exists(ce_path):
                    ce_data = np.load(ce_path)
                    subject_ce_labels.append(ce_data)
                    
            except Exception as e:
                print(f"Error processing {subject}-{scene} for mixed: {e}")
                continue
        
        # 合并所有场景的数据
        if subject_radar:
            combined_radar = np.concatenate(subject_radar, axis=0)
            combined_ecg = np.concatenate(subject_ecg, axis=0) if subject_ecg else None
            combined_ce_labels = np.concatenate(subject_ce_labels, axis=0) if subject_ce_labels else None
            
            # 创建受试者mixed目录
            subject_mixed_dir = os.path.join(self.output_dir, 'mixed', subject)
            os.makedirs(subject_mixed_dir, exist_ok=True)
            
            # 保存合并的数据
            np.save(os.path.join(subject_mixed_dir, 'radar_segments.npy'), combined_radar)
            
            if combined_ecg is not None:
                np.save(os.path.join(subject_mixed_dir, 'ecg_segments.npy'), combined_ecg)
                
            if combined_ce_labels is not None:
                np.save(os.path.join(subject_mixed_dir, 'ce_labels.npy'), combined_ce_labels)
            
            # 保存受试者元数据
            subject_meta = {
                'subject': subject,
                'scenes_included': list(set(meta['scene'] for meta in subject_metadata)),
                'total_segments': len(combined_radar),
                'segments_per_scene': {
                    scene: len([m for m in subject_metadata if m['scene'] == scene])
                    for scene in self.scenes
                },
                'data_shapes': {
                    'radar': combined_radar.shape,
                    'ecg': combined_ecg.shape if combined_ecg is not None else None,
                    'ce_labels': combined_ce_labels.shape if combined_ce_labels is not None else None
                },
                'paper_compliant': self.paper_compliant,
                'liu_compatible': self.liu_compatible,
                'created_at': datetime.now().isoformat()
            }
            
            # 保存详细元数据
            with open(os.path.join(subject_mixed_dir, 'metadata.json'), 'w') as f:
                json.dump(subject_meta, f, indent=2, default=str)
            
            with open(os.path.join(subject_mixed_dir, 'segments_info.json'), 'w') as f:
                json.dump(subject_metadata, f, indent=2, default=str)
            
            return subject_meta
        
        else:
            return {}

    def process_all_scenes_mixed(self) -> Dict[str, Any]:
        """
        处理所有受试者的mixed数据，按受试者组织。
        
        Returns:
            处理结果汇总
        """
        
        mixed_summary = {
            'subjects_processed': 0,
            'subjects_failed': 0,
            'total_subjects': len(self.subjects),
            'subjects_data': {},
            'overall_stats': {
                'total_radar_segments': 0,
                'total_ecg_segments': 0,
                'total_ce_labels': 0,
                'scenes_coverage': {scene: 0 for scene in self.scenes}
            }
        }
        
        # 确保mixed目录存在
        mixed_dir = os.path.join(self.output_dir, 'mixed')
        os.makedirs(mixed_dir, exist_ok=True)
        
        # 处理每个受试者
        for subject in self.subjects:
            try:
                print(f"Processing mixed data for {subject}...")
                subject_result = self.process_subject_mixed(subject)
                
                if subject_result:
                    mixed_summary['subjects_processed'] += 1
                    mixed_summary['subjects_data'][subject] = subject_result
                    
                    # 更新统计信息
                    mixed_summary['overall_stats']['total_radar_segments'] += subject_result['total_segments']
                    
                    if subject_result['data_shapes']['ecg'] is not None:
                        mixed_summary['overall_stats']['total_ecg_segments'] += subject_result['data_shapes']['ecg'][0]
                    
                    if subject_result['data_shapes']['ce_labels'] is not None:
                        mixed_summary['overall_stats']['total_ce_labels'] += subject_result['data_shapes']['ce_labels'][0]
                    
                    # 更新场景覆盖统计
                    for scene in subject_result['scenes_included']:
                        mixed_summary['overall_stats']['scenes_coverage'][scene] += 1
                
                else:
                    mixed_summary['subjects_failed'] += 1
                    
            except Exception as e:
                print(f"Error processing mixed data for {subject}: {e}")
                mixed_summary['subjects_failed'] += 1
        
        # 计算成功率
        mixed_summary['success_rate'] = (
            mixed_summary['subjects_processed'] / mixed_summary['total_subjects'] * 100
        )
        
        # 保存总体汇总信息
        mixed_summary['created_at'] = datetime.now().isoformat()
        mixed_summary['dataset_structure'] = 'subject_organized'
        mixed_summary['paper_compliant'] = self.paper_compliant
        mixed_summary['liu_compatible'] = self.liu_compatible
        
        # 保存mixed数据集总体信息
        with open(os.path.join(mixed_dir, 'dataset_summary.json'), 'w') as f:
            json.dump(mixed_summary, f, indent=2, default=str)
        
        # 创建简化的索引文件，便于快速查看数据集结构
        index_info = {
            'dataset_type': 'mixed_subject_organized',
            'total_subjects': mixed_summary['subjects_processed'],
            'subjects_list': list(mixed_summary['subjects_data'].keys()),
            'data_structure': 'mixed/{subject_id}/[radar_segments.npy, ecg_segments.npy, ce_labels.npy]',
            'usage_example': {
                'load_subject_data': "subject_dir = 'mixed/GDN0001/'; radar = np.load(subject_dir + 'radar_segments.npy')",
                'load_all_subjects': "subjects = ['GDN0001', 'GDN0002', ...]; data = [np.load(f'mixed/{s}/radar_segments.npy') for s in subjects]"
            }
        }
        
        with open(os.path.join(mixed_dir, 'index.json'), 'w') as f:
            json.dump(index_info, f, indent=2, default=str)
        
        return mixed_summary

    def validate_dataset(self) -> Dict[str, Any]:
        """验证数据集的完整性和结构。"""
        
        validation = {
            'total_expected': len(self.subjects) * len(self.scenes),
            'found_files': 0,
            'missing_files': [],
            'found_subjects': set(),
            'completeness': 0.0
        }
        
        # 检查原始.mat文件
        for subject in self.subjects:
            for scene in self.scenes:
                code = self.scene_codes[scene]
                filename = f"{subject}_{code}_{scene}.mat"
                mat_path = os.path.join(self.raw_dir, subject, filename)
                
                if os.path.exists(mat_path):
                    validation['found_files'] += 1
                    validation['found_subjects'].add(subject)
                else:
                    validation['missing_files'].append(mat_path)
        
        validation['completeness'] = (validation['found_files'] / validation['total_expected']) * 100
        return validation

    def validate_mixed_dataset(self) -> Dict[str, Any]:
        """验证mixed数据集的结构和完整性。"""
        
        mixed_dir = os.path.join(self.output_dir, 'mixed')
        validation = {
            'mixed_dir_exists': os.path.exists(mixed_dir),
            'subjects_found': [],
            'subjects_missing': [],
            'data_integrity': {},
            'structure_valid': True
        }
        
        if not validation['mixed_dir_exists']:
            validation['structure_valid'] = False
            return validation
        
        # 检查每个受试者的mixed数据
        for subject in self.subjects:
            subject_dir = os.path.join(mixed_dir, subject)
            
            if os.path.exists(subject_dir):
                validation['subjects_found'].append(subject)
                
                # 检查数据文件
                radar_path = os.path.join(subject_dir, 'radar_segments.npy')
                ecg_path = os.path.join(subject_dir, 'ecg_segments.npy')
                ce_path = os.path.join(subject_dir, 'ce_labels.npy')
                meta_path = os.path.join(subject_dir, 'metadata.json')
                
                subject_validation = {
                    'has_radar': os.path.exists(radar_path),
                    'has_ecg': os.path.exists(ecg_path),
                    'has_ce_labels': os.path.exists(ce_path),
                    'has_metadata': os.path.exists(meta_path),
                    'shapes': {}
                }
                
                # 验证数据形状
                try:
                    if subject_validation['has_radar']:
                        radar_data = np.load(radar_path)
                        subject_validation['shapes']['radar'] = radar_data.shape
                        
                    if subject_validation['has_ecg']:
                        ecg_data = np.load(ecg_path)
                        subject_validation['shapes']['ecg'] = ecg_data.shape
                        
                    if subject_validation['has_ce_labels']:
                        ce_data = np.load(ce_path)
                        subject_validation['shapes']['ce_labels'] = ce_data.shape
                        
                except Exception as e:
                    subject_validation['error'] = str(e)
                    validation['structure_valid'] = False
                
                validation['data_integrity'][subject] = subject_validation
            else:
                validation['subjects_missing'].append(subject)
        
        return validation

    def print_stats(self):
        """打印处理统计信息。"""
        print(f"Processing Statistics:")
        print(f"  - Files processed: {self.stats['processed']}")
        print(f"  - Files failed: {self.stats['failed']}")
        print(f"  - Total segments: {self.stats['total_segments']}")
        if self.stats['processed'] + self.stats['failed'] > 0:
            success_rate = self.stats['processed'] / (self.stats['processed'] + self.stats['failed']) * 100
            print(f"  - Success rate: {success_rate:.1f}%")

    def load_subject_mixed_data(self, subject: str) -> Dict[str, np.ndarray]:
        """
        加载特定受试者的mixed数据。
        
        Args:
            subject: 受试者ID (如 'GDN0001')
            
        Returns:
            包含radar、ecg、ce_labels数据的字典
        """
        
        subject_dir = os.path.join(self.output_dir, 'mixed', subject)
        
        if not os.path.exists(subject_dir):
            raise FileNotFoundError(f"Mixed data not found for subject {subject}")
        
        data = {}
        
        # 加载radar数据
        radar_path = os.path.join(subject_dir, 'radar_segments.npy')
        if os.path.exists(radar_path):
            data['radar'] = np.load(radar_path)
        
        # 加载ECG数据
        ecg_path = os.path.join(subject_dir, 'ecg_segments.npy')
        if os.path.exists(ecg_path):
            data['ecg'] = np.load(ecg_path)
        
        # 加载CE labels
        ce_path = os.path.join(subject_dir, 'ce_labels.npy')
        if os.path.exists(ce_path):
            data['ce_labels'] = np.load(ce_path)
        
        # 加载元数据
        meta_path = os.path.join(subject_dir, 'metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                data['metadata'] = json.load(f)
        
        return data

    def load_all_mixed_data(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        加载所有受试者的mixed数据。
        
        Returns:
            以受试者ID为键的嵌套字典
        """
        
        mixed_dir = os.path.join(self.output_dir, 'mixed')
        
        if not os.path.exists(mixed_dir):
            raise FileNotFoundError("Mixed dataset directory not found")
        
        all_data = {}
        
        # 获取所有受试者目录
        subject_dirs = [d for d in os.listdir(mixed_dir) 
                       if os.path.isdir(os.path.join(mixed_dir, d)) and d.startswith('GDN')]
        
        for subject in subject_dirs:
            try:
                all_data[subject] = self.load_subject_mixed_data(subject)
            except Exception as e:
                print(f"Error loading data for {subject}: {e}")
        
        return all_data

    def get_dataset_overview(self) -> Dict[str, Any]:
        """获取数据集概览信息。"""
        
        overview = {
            'scenes': {},
            'mixed': {},
            'total_stats': {
                'processed_scenes': 0,
                'total_segments': 0,
                'available_subjects': set()
            }
        }
        
        # 检查场景数据
        for scene in self.scenes:
            scene_dir = os.path.join(self.output_dir, scene)
            scene_info = {
                'exists': os.path.exists(scene_dir),
                'subjects': [],
                'total_segments': 0
            }
            
            if scene_info['exists']:
                overview['total_stats']['processed_scenes'] += 1
                subject_dirs = [d for d in os.listdir(scene_dir) 
                               if os.path.isdir(os.path.join(scene_dir, d)) and d.startswith('GDN')]
                
                for subject in subject_dirs:
                    radar_path = os.path.join(scene_dir, subject, 'radar_segments.npy')
                    if os.path.exists(radar_path):
                        radar_data = np.load(radar_path)
                        scene_info['subjects'].append(subject)
                        scene_info['total_segments'] += len(radar_data)
                        overview['total_stats']['available_subjects'].add(subject)
            
            overview['scenes'][scene] = scene_info
        
        # 检查mixed数据
        mixed_dir = os.path.join(self.output_dir, 'mixed')
        mixed_info = {
            'exists': os.path.exists(mixed_dir),
            'subjects': [],
            'total_segments': 0
        }
        
        if mixed_info['exists']:
            subject_dirs = [d for d in os.listdir(mixed_dir) 
                           if os.path.isdir(os.path.join(mixed_dir, d)) and d.startswith('GDN')]
            
            for subject in subject_dirs:
                radar_path = os.path.join(mixed_dir, subject, 'radar_segments.npy')
                if os.path.exists(radar_path):
                    radar_data = np.load(radar_path)
                    mixed_info['subjects'].append(subject)
                    mixed_info['total_segments'] += len(radar_data)
        
        overview['mixed'] = mixed_info
        overview['total_stats']['available_subjects'] = list(overview['total_stats']['available_subjects'])
        overview['total_stats']['total_segments'] = sum(s['total_segments'] for s in overview['scenes'].values())
        
        return overview


# Convenience functions
def quick_process_scene(scene: str, **kwargs) -> Dict[str, Any]:
    """Quick processing of a single scene with paper-compliant output."""
    loader = DatasetLoader(**kwargs)
    return loader.process_scene(scene)


def quick_process_mixed(**kwargs) -> Dict[str, Any]:
    """Quick processing of mixed dataset with paper-compliant output (subject-organized)."""
    loader = DatasetLoader(**kwargs)
    return loader.process_all_scenes_mixed()


def quick_load_subject_data(subject: str, **kwargs) -> Dict[str, np.ndarray]:
    """Quick loading of a subject's mixed data."""
    loader = DatasetLoader(**kwargs)
    return loader.load_subject_mixed_data(subject)


def check_paper_compliance(data_dir: str) -> Dict[str, bool]:
    """Check if existing processed data meets paper compliance requirements."""
    
    compliance = {
        'radar_exists': False,
        'ecg_exists': False,
        'ce_labels_exists': False,
        'radar_shape_correct': False,
        'ecg_shape_correct': False,
        'ce_labels_shape_correct': False,
        'all_compliant': False
    }
    
    try:
        # Check radar segments
        radar_path = os.path.join(data_dir, 'radar_segments.npy')
        if os.path.exists(radar_path):
            compliance['radar_exists'] = True
            radar_data = np.load(radar_path)
            compliance['radar_shape_correct'] = (
                len(radar_data.shape) == 3 and 
                radar_data.shape[1] == 1 and 
                radar_data.shape[2] in [256, 1000]
            )
        
        # Check ECG segments
        ecg_path = os.path.join(data_dir, 'ecg_segments.npy')
        if os.path.exists(ecg_path):
            compliance['ecg_exists'] = True
            ecg_data = np.load(ecg_path)
            compliance['ecg_shape_correct'] = (
                len(ecg_data.shape) == 3 and 
                ecg_data.shape[1] == 1 and 
                ecg_data.shape[2] in [256, 1000]
            )
        
        # Check CE labels
        ce_path = os.path.join(data_dir, 'ce_labels.npy')
        if os.path.exists(ce_path):
            compliance['ce_labels_exists'] = True
            ce_data = np.load(ce_path)
            compliance['ce_labels_shape_correct'] = (
                len(ce_data.shape) == 3 and 
                ce_data.shape[1] == 5 and 
                ce_data.shape[2] in [256, 1000]
            )
        
        # Overall compliance
        compliance['all_compliant'] = (
            compliance['radar_exists'] and 
            compliance['radar_shape_correct'] and
            (not compliance['ecg_exists'] or compliance['ecg_shape_correct'])
        )
        
    except Exception as e:
        print(f"Error checking compliance: {e}")
    
    return compliance


# Example usage and main execution
if __name__ == "__main__":
    """
    Main execution for dataset processing.
    """
    
    print("Radar2ECGNet Dataset Loader v3.2")
    print("Subject-organized mixed dataset processing")
    print("=" * 50)
    
    # Configuration
    config = {
        'raw_dir': "C:/Xiao/Datasets/24GHz_Clinical_Radar_ECG_Dataset",
        'output_dir': "C:/Xiao/Radar2ECGNet/dataset",
        'liu_compatible': True,      # Use 1000 samples per segment
        'generate_ce_labels': True,  # Generate PQRST labels
        'paper_compliant': True      # Add channel dimensions (n, 1, 1000)
    }
    
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create loader
    loader = DatasetLoader(**config)
    
    # Show dataset overview
    print("\n" + "="*50)
    print("Current Dataset Overview")
    print("="*50)
    
    overview = loader.get_dataset_overview()
    
    print(f"Scene-based data:")
    for scene, info in overview['scenes'].items():
        if info['exists']:
            print(f"  {scene}: {len(info['subjects'])} subjects, {info['total_segments']} segments")
        else:
            print(f"  {scene}: Not processed")
    
    print(f"Mixed data:")
    if overview['mixed']['exists']:
        print(f"  {len(overview['mixed']['subjects'])} subjects, {overview['mixed']['total_segments']} segments")
    else:
        print(f"  Not processed")
    
    # Option 1: Process single scene (for testing)
    print("\n" + "="*50)
    print("Option 1: Process single scene (Resting)")
    print("="*50)
    
    try:
        summary = loader.process_scene('Resting')
        print(f"Resting scene processed:")
        print(f"  - Subjects processed: {summary['processed_subjects']}/{summary['total_subjects']}")
        print(f"  - Total segments: {summary['total_segments']}")
        print(f"  - Success rate: {summary['success_rate']:.1f}%")
        
        # Check output
        test_subject_dir = os.path.join(config['output_dir'], 'Resting', 'GDN0001')
        if os.path.exists(test_subject_dir):
            compliance = check_paper_compliance(test_subject_dir)
            print(f"  - Paper compliant: {'Yes' if compliance['all_compliant'] else 'No'}")
            
            # Show actual data shapes
            radar_path = os.path.join(test_subject_dir, 'radar_segments.npy')
            ecg_path = os.path.join(test_subject_dir, 'ecg_segments.npy')
            ce_path = os.path.join(test_subject_dir, 'ce_labels.npy')
            
            if os.path.exists(radar_path):
                radar_data = np.load(radar_path)
                print(f"  - Radar shape: {radar_data.shape}")
                
            if os.path.exists(ecg_path):
                ecg_data = np.load(ecg_path)
                print(f"  - ECG shape: {ecg_data.shape}")
                
            if os.path.exists(ce_path):
                ce_data = np.load(ce_path)
                print(f"  - CE labels shape: {ce_data.shape}")
    
    except Exception as e:
        print(f"Error processing Resting scene: {e}")
    
    # Option 2: Process mixed dataset (subject-organized)
    print("\n" + "="*50)
    print("Option 2: Process subject-organized mixed dataset")
    print("="*50)
    
    user_input = input("Process subject-organized mixed dataset? This may take a while. (y/n): ").lower().strip()
    
    if user_input in ['y', 'yes']:
        try:
            summary = loader.process_all_scenes_mixed()
            
            if summary and summary['subjects_processed'] > 0:
                print("Subject-organized mixed dataset processed:")
                print(f"  - Subjects processed: {summary['subjects_processed']}/{summary['total_subjects']}")
                print(f"  - Subjects failed: {summary['subjects_failed']}")
                print(f"  - Success rate: {summary['success_rate']:.1f}%")
                print(f"  - Total radar segments: {summary['overall_stats']['total_radar_segments']}")
                print(f"  - Total ECG segments: {summary['overall_stats']['total_ecg_segments']}")
                print(f"  - Total CE labels: {summary['overall_stats']['total_ce_labels']}")
                
                print(f"\nScene coverage:")
                for scene, count in summary['overall_stats']['scenes_coverage'].items():
                    print(f"  - {scene}: {count} subjects")
                
                # Test loading a subject's data
                print(f"\nTesting data loading for GDN0001:")
                try:
                    test_data = loader.load_subject_mixed_data('GDN0001')
                    print(f"  - Loaded data keys: {list(test_data.keys())}")
                    if 'radar' in test_data:
                        print(f"  - Radar shape: {test_data['radar'].shape}")
                    if 'ecg' in test_data:
                        print(f"  - ECG shape: {test_data['ecg'].shape}")
                    if 'ce_labels' in test_data:
                        print(f"  - CE labels shape: {test_data['ce_labels'].shape}")
                        
                except Exception as e:
                    print(f"  - Error loading test data: {e}")
                
                print(f"\nDataset structure:")
                print(f"  mixed/")
                print(f"  ├── GDN0001/")
                print(f"  │   ├── radar_segments.npy")
                print(f"  │   ├── ecg_segments.npy")
                print(f"  │   ├── ce_labels.npy")
                print(f"  │   └── metadata.json")
                print(f"  ├── GDN0002/")
                print(f"  │   └── ...")
                print(f"  └── dataset_summary.json")
                
                print(f"\nUsage examples:")
                print(f"  # Load specific subject")
                print(f"  subject_data = loader.load_subject_mixed_data('GDN0001')")
                print(f"  radar = subject_data['radar']  # Shape: (n_segments, 1, 1000)")
                print(f"  ")
                print(f"  # Load all subjects")
                print(f"  all_data = loader.load_all_mixed_data()")
                print(f"  all_radar = [data['radar'] for data in all_data.values()]")
                    
            else:
                print("No data processed successfully")
                
        except Exception as e:
            print(f"Error processing subject-organized mixed dataset: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Mixed dataset processing skipped")
    
    # Show final statistics
    print("\n" + "="*50)
    print("Processing Statistics")
    print("="*50)
    loader.print_stats()
    
    # Dataset validation
    print("\n" + "="*50)
    print("Dataset Validation")
    print("="*50)
    
    validation = loader.validate_dataset()
    print(f"Raw dataset completeness: {validation['completeness']:.1f}%")
    print(f"Found files: {validation['found_files']}/{validation['total_expected']}")
    print(f"Available subjects: {len(validation['found_subjects'])}")
    
    # Mixed dataset validation
    mixed_validation = loader.validate_mixed_dataset()
    if mixed_validation['mixed_dir_exists']:
        print(f"Mixed dataset subjects: {len(mixed_validation['subjects_found'])}/{len(loader.subjects)}")
        print(f"Missing subjects: {len(mixed_validation['subjects_missing'])}")
        if mixed_validation['subjects_missing']:
            print(f"  {mixed_validation['subjects_missing'][:3]}...")  # Show first 3
    else:
        print("Mixed dataset: Not created")
    
    print("\nDataset loader execution completed!")