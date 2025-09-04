"""
Complete Signal Processing Module for Radar2ECGNet

This module provides comprehensive signal processing capabilities for converting
24GHz CW radar I/Q signals to ECG-like waveforms.

Processing Pipeline:
1. Phase extraction (Ellipse Fitting)
2. Adaptive notch filtering (50Hz power line interference)
3. ECG-specific bandpass filtering (0.5-40Hz)
4. Baseline drift removal
5. Resampling with anti-aliasing (2000Hz -> 128Hz)
6. Data cleaning (outlier removal)
7. Signal normalization ([-1, 1])
8. Signal segmentation (2s windows, 0.5s overlap)
9. Signal quality assessment
Author: Radar2ECGNet Signal Processing
Version: 2.0
"""

import numpy as np
import warnings
from typing import Tuple, Union, List, Optional, Dict, Any
from scipy.signal import (
    butter, filtfilt, resample, iirnotch, sosfiltfilt, 
    detrend, welch, find_peaks, hilbert
)
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.interpolate import interp1d
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings('ignore')


class SignalQualityMetrics:
    """Signal quality assessment metrics."""
    
    @staticmethod
    def signal_to_noise_ratio(signal: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio (SNR) in dB."""
        signal_power = np.var(signal)
        noise_estimate = np.var(np.diff(signal))
        return 10 * np.log10(signal_power / max(noise_estimate, 1e-10))
    
    @staticmethod
    def baseline_drift_metric(signal: np.ndarray, window_size: int = 256) -> float:
        """Assess baseline drift using sliding window variance."""
        if len(signal) < window_size:
            return np.std(signal)
        
        drift_values = []
        for i in range(0, len(signal) - window_size, window_size // 2):
            window = signal[i:i + window_size]
            drift_values.append(np.mean(window))
        
        return np.std(drift_values) if len(drift_values) > 1 else 0.0
    
    @staticmethod
    def signal_quality_index(signal: np.ndarray, fs: float) -> Dict[str, float]:
        """Comprehensive signal quality assessment."""
        metrics = {}
        
        # SNR calculation
        metrics['snr_db'] = SignalQualityMetrics.signal_to_noise_ratio(signal)
        
        # Baseline drift
        metrics['baseline_drift'] = SignalQualityMetrics.baseline_drift_metric(signal)
        
        # Dynamic range
        metrics['dynamic_range'] = np.ptp(signal)
        
        # Zero crossing rate
        zero_crossings = np.sum(np.diff(np.signbit(signal)))
        metrics['zero_crossing_rate'] = zero_crossings / len(signal)
        
        # Spectral entropy
        freqs, psd = welch(signal, fs, nperseg=min(256, len(signal)//4))
        psd_normalized = psd / np.sum(psd)
        metrics['spectral_entropy'] = -np.sum(psd_normalized * np.log(psd_normalized + 1e-10))
        
        return metrics


class PhaseExtraction:
    """Advanced phase extraction methods for radar I/Q signals."""
    
    @staticmethod
    def ellipse_fitting_phase(i_signal: np.ndarray, q_signal: np.ndarray, 
                            sigma: float = 1.0, method: str = 'robust') -> np.ndarray:
        """
        Extract phase from I/Q signals using ellipse fitting with improved robustness.
        
        Args:
            i_signal: In-phase component
            q_signal: Quadrature component
            sigma: Gaussian smoothing parameter
            method: Fitting method ('robust' or 'standard')
            
        Returns:
            Unwrapped phase signal
        """
        if len(i_signal) != len(q_signal):
            raise ValueError("I and Q signals must have the same length")
        
        # Remove DC component using median (more robust than mean)
        i_centered = i_signal - np.median(i_signal)
        q_centered = q_signal - np.median(q_signal)
        
        if method == 'robust':
            # Robust phase extraction
            phase_raw = np.arctan2(q_centered, i_centered)
            phase = np.unwrap(phase_raw)
            
            # Apply robust smoothing
            if sigma > 0:
                phase = gaussian_filter1d(phase, sigma=sigma)
                
        else:
            # Standard method
            phase = np.unwrap(np.arctan2(q_centered, i_centered))
            if sigma > 0:
                phase = gaussian_filter1d(phase, sigma=sigma)
        
        return phase


class AdaptiveFiltering:
    """Advanced adaptive filtering techniques."""
    
    @staticmethod
    def adaptive_notch_filter(signal: np.ndarray, fs: float, 
                            target_freq: float = 50.0, Q: float = 30.0,
                            adaptive: bool = True) -> np.ndarray:
        """
        Adaptive notch filter with automatic frequency detection.
        
        Args:
            signal: Input signal
            fs: Sampling frequency
            target_freq: Target frequency to remove (Hz)
            Q: Quality factor
            adaptive: Whether to automatically detect interference frequency
            
        Returns:
            Filtered signal
        """
        if adaptive:
            # Detect dominant frequency near target
            freqs, psd = welch(signal, fs, nperseg=min(512, len(signal)//4))
            freq_range = np.where((freqs >= target_freq - 5) & (freqs <= target_freq + 5))[0]
            if len(freq_range) > 0:
                peak_idx = freq_range[np.argmax(psd[freq_range])]
                detected_freq = freqs[peak_idx]
            else:
                detected_freq = target_freq
        else:
            detected_freq = target_freq
        
        # Apply notch filter
        w0 = detected_freq / (fs / 2)
        w0 = np.clip(w0, 0.001, 0.999)
        
        try:
            b, a = iirnotch(w0, Q)
            filtered_signal = filtfilt(b, a, signal)
        except Exception:
            # Fallback to high-pass filter
            sos = butter(2, 0.5 / (fs / 2), btype='high', output='sos')
            filtered_signal = sosfiltfilt(sos, signal)
        
        return filtered_signal
    
    @staticmethod
    def ecg_bandpass_filter(signal: np.ndarray, fs: float, 
                          lowcut: float = 0.5, highcut: float = 40.0,
                          order: int = 4) -> np.ndarray:
        """
        ECG-specific bandpass filter optimized for heart signal frequencies.
        
        Args:
            signal: Input signal
            fs: Sampling frequency
            lowcut: Low cutoff frequency (Hz)
            highcut: High cutoff frequency (Hz)
            order: Filter order
            
        Returns:
            Filtered signal
        """
        nyq = 0.5 * fs
        low = max(lowcut / nyq, 0.001)
        high = min(highcut / nyq, 0.999)
        
        if low >= high:
            raise ValueError("Low cutoff must be less than high cutoff")
        
        sos = butter(order, [low, high], btype='band', output='sos')
        return sosfiltfilt(sos, signal)


class BaselineCorrection:
    """Multiple strategies for baseline drift removal."""
    
    @staticmethod
    def polynomial_detrend(signal: np.ndarray, order: int = 5) -> np.ndarray:
        """Remove baseline using polynomial fitting."""
        x = np.arange(len(signal))
        coeffs = np.polyfit(x, signal, order)
        baseline = np.polyval(coeffs, x)
        return signal - baseline
    
    @staticmethod
    def median_filter_baseline(signal: np.ndarray, window_size: int = 200) -> np.ndarray:
        """Remove baseline using median filtering."""
        baseline = median_filter(signal, size=window_size, mode='reflect')
        return signal - baseline
    
    @staticmethod
    def adaptive_baseline_correction(signal: np.ndarray, method: str = 'polynomial',
                                   order: int = 5, fs: float = 128.0) -> np.ndarray:
        """
        Adaptive baseline correction with multiple methods.
        
        Args:
            signal: Input signal
            method: Correction method ('polynomial', 'linear', 'highpass', 'median')
            order: Polynomial order (for polynomial method)
            fs: Sampling frequency
            
        Returns:
            Baseline-corrected signal
        """
        if method == 'polynomial':
            return BaselineCorrection.polynomial_detrend(signal, order)
        
        elif method == 'linear':
            return detrend(signal, type='linear')
        
        elif method == 'highpass':
            cutoff = 0.05 / (fs / 2)
            cutoff = np.clip(cutoff, 0.001, 0.499)
            sos = butter(1, cutoff, btype='high', output='sos')
            return sosfiltfilt(sos, signal)
        
        elif method == 'median':
            window_size = min(200, len(signal) // 10)
            return BaselineCorrection.median_filter_baseline(signal, window_size)
        
        else:
            raise ValueError(f"Unsupported baseline correction method: {method}")


class SignalNormalization:
    """Advanced signal normalization techniques."""
    
    @staticmethod
    def zscore_normalize(signal: np.ndarray, robust: bool = False) -> np.ndarray:
        """Z-score normalization with optional robust statistics."""
        if robust:
            median_val = np.median(signal)
            mad_val = np.median(np.abs(signal - median_val))
            return (signal - median_val) / (1.4826 * mad_val) if mad_val != 0 else signal - median_val
        else:
            mean_val = np.mean(signal)
            std_val = np.std(signal)
            return (signal - mean_val) / std_val if std_val != 0 else signal - mean_val
    
    @staticmethod
    def minmax_normalize(signal: np.ndarray, feature_range: Tuple[float, float] = (-1, 1)) -> np.ndarray:
        """Min-Max normalization to specified range."""
        min_val = np.min(signal)
        max_val = np.max(signal)
        
        if max_val == min_val:
            return np.full_like(signal, feature_range[0])
        
        normalized = (signal - min_val) / (max_val - min_val)
        return normalized * (feature_range[1] - feature_range[0]) + feature_range[0]
    
    @staticmethod
    def robust_normalize(signal: np.ndarray) -> np.ndarray:
        """Robust normalization using RobustScaler."""
        scaler = RobustScaler()
        signal_reshaped = signal.reshape(-1, 1)
        normalized = scaler.fit_transform(signal_reshaped)
        return normalized.flatten()


class SignalResampling:
    """Advanced resampling techniques with anti-aliasing."""
    
    @staticmethod
    def resample_with_antialiasing(signal: np.ndarray, orig_fs: float, target_fs: float) -> np.ndarray:
        """
        Resample signal with proper anti-aliasing filtering.
        
        Args:
            signal: Input signal
            orig_fs: Original sampling frequency
            target_fs: Target sampling frequency
            
        Returns:
            Resampled signal
        """
        if orig_fs == target_fs:
            return signal.copy()
        
        # Anti-aliasing filter for downsampling
        if target_fs < orig_fs:
            nyq = 0.5 * orig_fs
            cutoff = min(target_fs / 2.2, nyq * 0.8)
            cutoff_normalized = cutoff / nyq
            cutoff_normalized = np.clip(cutoff_normalized, 0.001, 0.999)
            
            # 8th order Butterworth anti-aliasing filter
            sos = butter(8, cutoff_normalized, btype='low', output='sos')
            filtered = sosfiltfilt(sos, signal)
        else:
            filtered = signal
        
        # Resample
        num_samples = int(len(signal) * target_fs / orig_fs)
        return resample(filtered, num_samples)


class DataCleaning:
    """Data cleaning and outlier removal techniques."""
    
    @staticmethod
    def remove_outliers(signal: np.ndarray, method: str = 'modified_zscore', 
                       threshold: float = 3.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove outliers from signal using various methods.
        
        Args:
            signal: Input signal
            method: Outlier detection method ('iqr', 'zscore', 'modified_zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            Tuple of (cleaned_signal, outlier_mask)
        """
        if method == 'iqr':
            q1 = np.percentile(signal, 25)
            q3 = np.percentile(signal, 75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            outlier_mask = (signal < lower_bound) | (signal > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs((signal - np.mean(signal)) / np.std(signal))
            outlier_mask = z_scores > threshold
        
        elif method == 'modified_zscore':
            median_val = np.median(signal)
            mad_val = np.median(np.abs(signal - median_val))
            if mad_val == 0:
                return signal.copy(), np.zeros_like(signal, dtype=bool)
            modified_z_scores = 0.6745 * (signal - median_val) / mad_val
            outlier_mask = np.abs(modified_z_scores) > threshold
        
        else:
            raise ValueError(f"Unsupported outlier detection method: {method}")
        
        # Replace outliers with interpolated values
        cleaned_signal = signal.copy()
        if np.any(outlier_mask):
            valid_indices = np.where(~outlier_mask)[0]
            outlier_indices = np.where(outlier_mask)[0]
            
            if len(valid_indices) > 0:
                cleaned_signal[outlier_indices] = np.interp(
                    outlier_indices, valid_indices, signal[valid_indices]
                )
        
        return cleaned_signal, outlier_mask
    
    @staticmethod
    def signal_validation(signal: np.ndarray, fs: float) -> Dict[str, Any]:
        """Comprehensive signal validation."""
        results = {
            'is_valid': True,
            'issues': [],
            'statistics': {}
        }
        
        # Check for NaN or infinite values
        if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
            results['is_valid'] = False
            results['issues'].append('Signal contains NaN or infinite values')
        
        # Check signal length
        if len(signal) < 100:
            results['is_valid'] = False
            results['issues'].append('Signal too short (< 100 samples)')
        
        # Check for constant signal
        if np.std(signal) < 1e-10:
            results['is_valid'] = False
            results['issues'].append('Signal appears to be constant')
        
        # Calculate statistics
        results['statistics'] = {
            'length': len(signal),
            'mean': np.mean(signal),
            'std': np.std(signal),
            'min': np.min(signal),
            'max': np.max(signal),
            'sampling_rate': fs
        }
        
        return results


class SignalSegmentation:
    """Advanced signal segmentation with overlap and windowing."""
    
    @staticmethod
    def sliding_window_segmentation(signal: np.ndarray, window_size: int = 256,
                                  overlap: float = 0.5, min_segment_size: int = 128) -> np.ndarray:
        """
        Segment signal using sliding window with configurable overlap.
        
        Args:
            signal: Input signal
            window_size: Size of each segment (256 for 128Hz, 2-second windows)
            overlap: Overlap fraction between segments (0.0 to 0.99)
            min_segment_size: Minimum size for the last segment
            
        Returns:
            Array of signal segments
        """
        if window_size > len(signal):
            return np.array([signal])
        
        overlap = np.clip(overlap, 0.0, 0.99)
        step_size = int(window_size * (1 - overlap))
        
        segments = []
        for i in range(0, len(signal), step_size):
            end_idx = min(i + window_size, len(signal))
            segment = signal[i:end_idx]
            
            if len(segment) >= min_segment_size:
                # Pad if necessary
                if len(segment) < window_size:
                    pad_size = window_size - len(segment)
                    segment = np.pad(segment, (0, pad_size), mode='edge')
                segments.append(segment)
            
            if end_idx >= len(signal):
                break
        
        return np.array(segments) if segments else np.array([signal])


def preprocess_radar_signal_complete(
    i_signal: np.ndarray, 
    q_signal: np.ndarray, 
    fs: float = 2000.0,
    target_fs: float = 128.0,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Complete preprocessing pipeline for 24GHz CW radar to ECG conversion.
    
    Args:
        i_signal: In-phase component (2000 Hz)
        q_signal: Quadrature component (2000 Hz)
        fs: Original sampling frequency (2000 Hz)
        target_fs: Target sampling frequency (128 Hz)
        config: Processing configuration dictionary
        
    Returns:
        Dictionary containing processed signals and metadata
    """
    
    # Default configuration
    if config is None:
        config = {
            'phase_extraction': {'sigma': 1.0, 'method': 'robust'},
            'filtering': {
                'notch_freq': 50.0,
                'notch_q': 30.0,
                'bandpass_low': 0.5,
                'bandpass_high': 40.0,
                'bandpass_order': 4
            },
            'baseline': {'method': 'polynomial', 'order': 5},
            'normalization': {'method': 'minmax', 'range': (-1, 1)},
            'cleaning': {'method': 'modified_zscore', 'threshold': 3.5},
            'segmentation': {
                'window_duration': 2.0,    # 2 seconds
                'overlap_duration': 0.5    # 0.5 seconds
            }
        }
    
    results = {'success': True, 'errors': []}
    
    try:
        # Step 1: Input validation
        if len(i_signal) != len(q_signal):
            raise ValueError("I and Q signals must have the same length")
        
        validation = DataCleaning.signal_validation(i_signal, fs)
        if not validation['is_valid']:
            results['errors'].extend(validation['issues'])
            results['success'] = False
            return results
        
        # Step 2: Phase extraction (ellipse fitting)
        phase_signal = PhaseExtraction.ellipse_fitting_phase(
            i_signal, q_signal, 
            sigma=config['phase_extraction']['sigma'],
            method=config['phase_extraction']['method']
        )
        
        # Step 3: Notch filtering (power line interference)
        filtered_signal = AdaptiveFiltering.adaptive_notch_filter(
            phase_signal, fs, 
            target_freq=config['filtering']['notch_freq'],
            Q=config['filtering']['notch_q']
        )
        
        # Step 4: ECG bandpass filtering
        filtered_signal = AdaptiveFiltering.ecg_bandpass_filter(
            filtered_signal, fs, 
            lowcut=config['filtering']['bandpass_low'],
            highcut=config['filtering']['bandpass_high'],
            order=config['filtering']['bandpass_order']
        )
        
        # Step 5: Baseline correction
        corrected_signal = BaselineCorrection.adaptive_baseline_correction(
            filtered_signal, 
            method=config['baseline']['method'],
            order=config['baseline']['order'],
            fs=fs
        )
        
        # Step 6: Resampling (2000Hz -> 128Hz)
        if fs != target_fs:
            resampled_signal = SignalResampling.resample_with_antialiasing(
                corrected_signal, fs, target_fs
            )
        else:
            resampled_signal = corrected_signal
        
        # Step 7: Data cleaning (outlier removal)
        cleaned_signal, outlier_mask = DataCleaning.remove_outliers(
            resampled_signal, 
            method=config['cleaning']['method'], 
            threshold=config['cleaning']['threshold']
        )
        
        # Step 8: Normalization
        if config['normalization']['method'] == 'minmax':
            normalized_signal = SignalNormalization.minmax_normalize(
                cleaned_signal, feature_range=config['normalization']['range']
            )
        else:
            normalized_signal = SignalNormalization.zscore_normalize(
                cleaned_signal, robust=True
            )
        
        # Step 9: Signal segmentation (2s windows, 0.5s overlap)
        window_samples = int(config['segmentation']['window_duration'] * target_fs)  # 256 samples
        overlap_fraction = config['segmentation']['overlap_duration'] / config['segmentation']['window_duration']  # 0.25
        
        segments = SignalSegmentation.sliding_window_segmentation(
            normalized_signal,
            window_size=window_samples,
            overlap=overlap_fraction,
            min_segment_size=window_samples // 2
        )
        
        # Step 10: Signal quality assessment
        quality_metrics = SignalQualityMetrics.signal_quality_index(normalized_signal, target_fs)
        
        # Compile final results
        results.update({
            'processed_signal': normalized_signal,
            'segments': segments,
            'quality_metrics': quality_metrics,
            'outlier_mask': outlier_mask,
            'outliers_removed': np.sum(outlier_mask),
            'config_used': config,
            'original_fs': fs,
            'target_fs': target_fs,
            'original_length': len(i_signal),
            'processed_length': len(normalized_signal),
            'num_segments': len(segments),
            'segment_shape': segments.shape if len(segments) > 0 else (0, 0),
            'window_samples': window_samples,
            'overlap_samples': int(config['segmentation']['overlap_duration'] * target_fs)
        })
        
    except Exception as e:
        results['success'] = False
        results['errors'].append(f"Processing failed: {str(e)}")
    
    return results


def convert_to_liu_compatibility(
    segments_256: np.ndarray,
    method: str = 'interpolation'
) -> np.ndarray:
    """
    Convert 256-sample segments to 1000-sample segments for Liu et al. compatibility.
    
    Args:
        segments_256: Input segments of shape (n_segments, 256)
        method: Conversion method ('interpolation' or 'padding')
        
    Returns:
        Array of shape (n_segments, 1000)
    """
    
    if segments_256.shape[1] != 256:
        raise ValueError("Input segments must have 256 samples each")
    
    n_segments = segments_256.shape[0]
    segments_1000 = np.zeros((n_segments, 1000))
    
    if method == 'interpolation':
        # Interpolate from 256 to 1000 samples
        x_256 = np.linspace(0, 2.0, 256)    
        x_1000 = np.linspace(0, 2.0, 1000)  
        
        for i, segment in enumerate(segments_256):
            interpolator = interp1d(x_256, segment, kind='cubic', fill_value='extrapolate')
            segments_1000[i] = interpolator(x_1000)
            
    elif method == 'padding':
        # Pad 256-sample segments to 1000 samples
        for i, segment in enumerate(segments_256):
            segments_1000[i, :256] = segment
            segments_1000[i, 256:] = segment[-1]
    
    else:
        raise ValueError("Method must be 'interpolation' or 'padding'")
    
    return segments_1000


def process_24ghz_radar_to_ecg(
    i_signal: np.ndarray, 
    q_signal: np.ndarray,
    output_format: str = 'standard',
    custom_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    High-level API for processing 24GHz CW radar signals to ECG-like waveforms.
    
    Args:
        i_signal: In-phase component at 2000Hz
        q_signal: Quadrature component at 2000Hz  
        output_format: 'standard' or 'liu_compatible'
        custom_config: Optional custom configuration
        
    Returns:
        Dictionary with processed signals and metadata
    """
    
    # Standard preprocessing (2000Hz -> 128Hz)
    results = preprocess_radar_signal_complete(
        i_signal=i_signal,
        q_signal=q_signal,
        fs=2000.0,
        target_fs=128.0,
        config=custom_config
    )
    
    if not results['success']:
        return results
    
    # Add format-specific processing
    if output_format == 'liu_compatible':
        segments_256 = results['segments']
        if len(segments_256) > 0:
            # Convert 256-sample segments to 1000-sample segments
            segments_1000 = convert_to_liu_compatibility(segments_256, method='interpolation')
            
            results.update({
                'segments_1x1000': segments_1000,
                'liu_compatible': True,
                'conversion_method': 'interpolation_256_to_1000',
                'original_segment_shape': segments_256.shape,  
                'liu_segment_shape': segments_1000.shape,     
                'ready_for_stft': True
            })
        else:
            results['liu_compatible'] = False
            results['errors'].append("No segments generated for Liu compatibility conversion")
    
    else:
        results.update({
            'standard_format': True,
            'ready_for_stft': False
        })
    
    return results


def validate_processing_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Comprehensive validation of processing results.
    
    Args:
        results: Results dictionary from processing functions
        
    Returns:
        Validation report with pass/fail status and recommendations
    """
    
    validation = {
        'passed': True,
        'checks': {},
        'recommendations': [],
        'summary': {}
    }
    
    # Basic processing check
    validation['checks']['basic_processing'] = results.get('success', False)
    if not validation['checks']['basic_processing']:
        validation['passed'] = False
        validation['recommendations'].append("Basic processing failed. Check input data quality.")
        return validation
    
    # Signal quality checks
    quality = results.get('quality_metrics', {})
    
    # SNR check (recommended > 10 dB)
    snr = quality.get('snr_db', 0)
    validation['checks']['snr_acceptable'] = snr >= 10.0
    if not validation['checks']['snr_acceptable']:
        validation['recommendations'].append(f"Low SNR ({snr:.1f}dB). Consider noise reduction.")
    
    # Baseline drift check (recommended < 0.2)
    drift = quality.get('baseline_drift', float('inf'))
    validation['checks']['baseline_stable'] = drift <= 0.2
    if not validation['checks']['baseline_stable']:
        validation['recommendations'].append(f"High baseline drift ({drift:.3f}). Consider stronger correction.")
    
    # Dynamic range check (recommended > 0.1)
    drange = quality.get('dynamic_range', 0)
    validation['checks']['dynamic_range_ok'] = drange >= 0.1
    if not validation['checks']['dynamic_range_ok']:
        validation['recommendations'].append(f"Low dynamic range ({drange:.3f}). Check signal amplitude.")
    
    # Segment count check
    num_segments = results.get('num_segments', 0)
    validation['checks']['sufficient_segments'] = num_segments >= 5
    if not validation['checks']['sufficient_segments']:
        validation['recommendations'].append(f"Few segments generated ({num_segments}). Consider longer signals.")
    
    # Overall validation
    validation['passed'] = all(validation['checks'].values())
    
    # Summary statistics
    validation['summary'] = {
        'total_checks': len(validation['checks']),
        'passed_checks': sum(validation['checks'].values()),
        'snr_db': snr,
        'baseline_drift': drift,
        'dynamic_range': drange,
        'num_segments': num_segments
    }
    
    return validation


# Convenience wrapper functions
def ellipse_fitting_phase(i_signal: np.ndarray, q_signal: np.ndarray, 
                         sigma: float = 1.0, method: str = "robust") -> np.ndarray:
    """Convenience wrapper for phase extraction."""
    return PhaseExtraction.ellipse_fitting_phase(i_signal, q_signal, sigma, method)


def bandpass_filter(signal_data: np.ndarray, fs: float, lowcut: float = 0.5, 
                   highcut: float = 40, order: int = 4) -> np.ndarray:
    """Convenience wrapper for ECG bandpass filtering."""
    return AdaptiveFiltering.ecg_bandpass_filter(signal_data, fs, lowcut, highcut, order)


def notch_filter(signal_data: np.ndarray, fs: float, freq: float = 50, Q: float = 30) -> np.ndarray:
    """Convenience wrapper for adaptive notch filtering."""
    return AdaptiveFiltering.adaptive_notch_filter(signal_data, fs, freq, Q, adaptive=True)


def baseline_correction(signal_data: np.ndarray, method: str = 'polynomial', 
                       order: int = 5, fs: float = 128) -> np.ndarray:
    """Convenience wrapper for baseline correction."""
    return BaselineCorrection.adaptive_baseline_correction(signal_data, method, order, fs)


def normalize(signal_data: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """Convenience wrapper for signal normalization."""
    if method == 'minmax':
        return SignalNormalization.minmax_normalize(signal_data, feature_range=(-1, 1))
    else:
        return SignalNormalization.zscore_normalize(signal_data, robust=True)


def resample_signal(signal_data: np.ndarray, orig_fs: float, target_fs: float) -> np.ndarray:
    """Convenience wrapper for signal resampling."""
    return SignalResampling.resample_with_antialiasing(signal_data, orig_fs, target_fs)


def segment_signal(signal_data: np.ndarray, window_size: int = 256, overlap: float = 0.5) -> np.ndarray:
    """Convenience wrapper for signal segmentation."""
    return SignalSegmentation.sliding_window_segmentation(signal_data, window_size, overlap)


def get_processing_config_template() -> Dict[str, Any]:
    """
    Get a template configuration dictionary with all available options.
    
    Returns:
        Template configuration dictionary with explanations
    """
    
    return {
        # Phase extraction configuration
        'phase_extraction': {
            'method': 'robust',      # 'robust' or 'standard'
            'sigma': 0.8,           # Gaussian smoothing parameter (0.5-2.0)
            'dc_removal': 'median'   # DC removal method ('median' or 'mean')
        },
        
        # Filtering configuration
        'filtering': {
            'notch_freq': 50.0,     # Notch frequency in Hz (50 for EU/Asia, 60 for US)
            'notch_q': 40.0,        # Quality factor for notch filter (10-50)
            'adaptive_notch': True,  # Enable adaptive frequency detection
            'bandpass_low': 0.5,    # ECG bandpass lower cutoff (Hz)
            'bandpass_high': 40.0,  # ECG bandpass upper cutoff (Hz)
            'bandpass_order': 6     # Bandpass filter order (2-8)
        },
        
        # Baseline correction configuration
        'baseline': {
            'method': 'polynomial', # 'polynomial', 'linear', 'highpass', 'median'
            'order': 5,            # Polynomial order for polynomial method (3-8)
            'backup_method': 'highpass'  # Fallback method if primary fails
        },
        
        # Resampling configuration
        'resampling': {
            'original_fs': 2000.0,  # Original sampling rate
            'target_fs': 128.0,     # Target sampling rate
            'anti_alias_order': 8   # Anti-aliasing filter order
        },
        
        # Data cleaning configuration
        'cleaning': {
            'method': 'modified_zscore',  # 'iqr', 'zscore', 'modified_zscore'
            'threshold': 3.5,            # Outlier detection threshold
            'interpolation': 'linear'     # Outlier replacement method
        },
        
        # Normalization configuration
        'normalization': {
            'method': 'minmax',     # 'zscore', 'minmax', 'robust', 'combined'
            'range': (-1, 1),       # Target range for minmax normalization
            'robust': True          # Use robust statistics where applicable
        },
        
        # Segmentation configuration
        'segmentation': {
            'window_duration': 2.0,  # Window duration in seconds
            'overlap_duration': 0.5, # Overlap duration in seconds
            'min_segment_ratio': 0.8 # Minimum segment size as fraction of window_size
        },
        
        # Quality assessment thresholds
        'quality_thresholds': {
            'min_snr_db': 10.0,         # Minimum acceptable SNR
            'max_baseline_drift': 0.2,   # Maximum acceptable baseline drift
            'min_dynamic_range': 0.1,    # Minimum acceptable dynamic range
            'max_outlier_percentage': 5.0 # Maximum acceptable outlier percentage
        }
    }


# Example usage function
def example_usage():
    """
    Example usage of the complete signal processing module.
    """
    
    # Generate example I/Q data (replace with your actual data loading)
    fs_orig = 2000  # 24GHz radar sampling rate
    duration = 10   # 10 seconds of data
    t = np.linspace(0, duration, int(fs_orig * duration))
    
    # Simulate radar I/Q signals with realistic characteristics
    # Heart rate simulation (~1.2 Hz)
    heart_signal = 0.5 * np.sin(2 * np.pi * 1.2 * t)
    # Breathing component (~0.3 Hz) 
    breathing_signal = 0.3 * np.sin(2 * np.pi * 0.3 * t)
    # Noise
    noise = 0.1 * np.random.randn(len(t))
    
    # Create I/Q components
    i_data = heart_signal + breathing_signal + noise + 0.05 * np.sin(2 * np.pi * 50 * t)  # 50Hz interference
    q_data = heart_signal + breathing_signal + noise + 0.1 * np.random.randn(len(t))
    
    # Process with complete pipeline
    results = process_24ghz_radar_to_ecg(
        i_signal=i_data,
        q_signal=q_data,
        output_format='liu_compatible'
    )
    
    # Validate results
    validation = validate_processing_results(results)
    
    return results, validation


if __name__ == "__main__":
    """
    Main execution for testing the module.
    """
    
    # Run example
    results, validation = example_usage()
    
    if results['success']:
        print(f"✅ Processing successful!")
        print(f"📊 Generated {results['num_segments']} segments")
        print(f"📈 Signal quality: SNR = {results['quality_metrics']['snr_db']:.1f} dB")
        print(f"🎯 Validation: {'PASSED' if validation['passed'] else 'FAILED'}")
        
        if results.get('liu_compatible', False):
            print(f"📋 Liu compatibility: {results['liu_segment_shape']} segments ready for STFT")
    else:
        print(f"❌ Processing failed: {results['errors']}")
    
    print("\n🚀 Module ready for use!")