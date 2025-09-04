"""
PyTorch Dataset Module for Radar2ECGNet

This module provides PyTorch Dataset and DataLoader functionality for training
the radar-to-ECG neural network models. It supports loading radar, ECG,
and optional cardiac event (CE) labels.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Tuple, Optional, Dict, Any, List
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiScenarioDataset(Dataset):
    """
    A powerful and unified Dataset class for handling multiple scenarios and subjects.
    It loads radar, ECG, and optionally CE labels from a nested directory structure.
    """
    def __init__(self, data_root: str, scenarios: List[str], subjects: Optional[List[str]] = None):
        self.data_root = data_root
        self.scenarios = scenarios
        self.subjects = subjects
        
        self.radar_segments = []
        self.ecg_segments = []
        self.ce_labels_segments = []
        self.metadata = []
        self.has_ce_labels = False
        
        self._load_all_data()
        
        if not self.radar_segments:
            raise RuntimeError(f"No data found for scenarios '{self.scenarios}' in '{self.data_root}'. Please check paths and data existence.")

        self.radar_segments = np.concatenate(self.radar_segments, axis=0)
        self.ecg_segments = np.concatenate(self.ecg_segments, axis=0)
        if self.ce_labels_segments:
            self.ce_labels_segments = np.concatenate(self.ce_labels_segments, axis=0)
            self.has_ce_labels = True

        logger.info(f"Multi-scenario dataset loaded for scenarios: {scenarios}")
        logger.info(f"  Total samples: {len(self)}")
        if self.has_ce_labels:
            logger.info(f"  Data loaded: Radar, ECG, CE Labels")
            
    
    def _load_all_data(self):
        """Load data from all specified scenarios and subjects."""
        for scenario in self.scenarios:
            scenario_dir = os.path.join(self.data_root, scenario)
            if not os.path.isdir(scenario_dir):
                logger.warning(f"Scenario directory not found: {scenario_dir}")
                continue
            
            subject_dirs = self.subjects if self.subjects is not None else sorted([d for d in os.listdir(scenario_dir) if os.path.isdir(os.path.join(scenario_dir, d))])
            
            for subject_dir in subject_dirs:
                subject_path = os.path.join(scenario_dir, subject_dir)
                radar_path = os.path.join(subject_path, 'radar_segments.npy')
                ecg_path = os.path.join(subject_path, 'ecg_segments.npy')
                
                if os.path.exists(radar_path) and os.path.exists(ecg_path):
                    self.radar_segments.append(np.load(radar_path))
                    self.ecg_segments.append(np.load(ecg_path))
                    
                    ce_labels_path = os.path.join(subject_path, 'ce_labels.npy')
                    if os.path.exists(ce_labels_path):
                        self.ce_labels_segments.append(np.load(ce_labels_path))
                    
                    num_segments = len(self.radar_segments[-1])
                    for i in range(num_segments):
                        self.metadata.append({
                            'scenario': scenario,
                            'subject': subject_dir
                        })
    
    def __len__(self) -> int:
        return len(self.radar_segments)
    
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = {
            'radar': torch.from_numpy(self.radar_segments[idx]).float(),
            'ecg': torch.from_numpy(self.ecg_segments[idx]).float(),
            'metadata': self.metadata[idx]
        }
        if self.has_ce_labels:
            sample['ce_labels'] = torch.from_numpy(self.ce_labels_segments[idx]).float()
        return sample
    
    
def create_data_loaders(
    dataset: Dataset,
    batch_size: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    num_workers: int = 4,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders from a single dataset.
    """
    torch.manual_seed(random_seed)
    
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    logger.info(f"Data loaders created from a dataset of {total_size} samples:")
    logger.info(f"  Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples | Test: {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader

def create_subject_specific_loaders(
    data_root: str,
    scenario: str,
    batch_size: int,
    train_subjects: List[str],
    val_subjects: List[str],
    test_subjects: List[str],
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders with subject-level splits for a specific scenario.
    """
    train_dataset = MultiScenarioDataset(data_root, scenarios=[scenario], subjects=train_subjects)
    val_dataset = MultiScenarioDataset(data_root, scenarios=[scenario], subjects=val_subjects)
    test_dataset = MultiScenarioDataset(data_root, scenarios=[scenario], subjects=test_subjects)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    logger.info(f"Subject-specific data loaders created for scenario '{scenario}':")
    logger.info(f"  Train: {len(train_dataset)} samples ({len(train_subjects)} subjects)")
    logger.info(f"  Val: {len(val_dataset)} samples ({len(val_subjects)} subjects)")
    logger.info(f"  Test: {len(test_dataset)} samples ({len(test_subjects)} subjects)")

    return train_loader, val_loader, test_loader

# --- Example Usage Functions ---

def load_mixed_dataset(data_root: str, batch_size: int, **kwargs) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    加载 'mixed' 数据集，并确保排除了已知损坏的受试者。
    """
    mixed_dir = os.path.join(data_root, 'mixed')
    
    # 1. 获取 'mixed' 目录下的所有受试者文件夹
    all_subjects_in_mixed = sorted([
        d for d in os.listdir(mixed_dir) 
        if os.path.isdir(os.path.join(mixed_dir, d))
    ])

    # 2. 定义并排除已知有问题的受试者
    known_bad_subjects = {'GDN0003', 'GDN0009'}
    subjects_to_load = [s for s in all_subjects_in_mixed if s not in known_bad_subjects]
    
    logging.info(f"在 'mixed' 数据集中，将加载 {len(subjects_to_load)} 位受试者的数据。")

    # 3. 使用筛选后的受试者列表来创建数据集
    dataset = MultiScenarioDataset(
        data_root=data_root,
        scenarios=['mixed'],
        subjects=subjects_to_load
    )
    
    return create_data_loaders(dataset, batch_size=batch_size, **kwargs)

def load_scenario_dataset(data_root: str, scenario: str, batch_size: int, **kwargs) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads a single scenario and splits the data by pre-defined subject lists.
    You can customize the subject lists here as needed for your experiments.
    """
    # Example subject splits for a dataset of 30 subjects
    
    bad_subjects = {'GDN0003', 'GDN0009'}
    
    train_subjects = [f'GDN{i:04d}' for i in range(1, 23) if f'GDN{i:04d}' not in bad_subjects]  # ~70% for training
    val_subjects = [f'GDN{i:04d}' for i in range(23, 27) if f'GDN{i:04d}' not in bad_subjects]   # ~15% for validation
    test_subjects = [f'GDN{i:04d}' for i in range(27, 31) if f'GDN{i:04d}' not in bad_subjects]  # ~15% for testing
    
    return create_subject_specific_loaders(
        data_root=data_root,
        scenario=scenario,
        batch_size=batch_size,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subjects=test_subjects,
        **kwargs
    )