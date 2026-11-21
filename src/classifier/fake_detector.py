"""
AI-Generated Video (Deepfake) Detector using Remote Photoplethysmography (rPPG).
Analyzes biological consistency of extracted pulse signals:
1. Cardiac frequency peak & SNR in physiological band (0.75 - 3.0 Hz / 45 - 180 bpm)
2. Spatial cross-correlation between Forehead and Cheek rPPG signals
3. Spectral entropy and temporal periodicity
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import scipy.signal as signal


def compute_pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Computes Pearson correlation coefficient between two 1D numpy arrays."""
    if len(x) < 2 or len(y) < 2:
        return 0.0
    r = np.corrcoef(x, y)[0, 1]
    return float(r) if not np.isnan(r) else 0.0


def compute_autocorrelation_periodicity(sig: np.ndarray, fs: float = 30.0) -> float:
    """Computes the maximum secondary peak of autocorrelation in cardiac lag range [0.4s - 1.5s]."""
    if len(sig) < int(fs * 2):
        return 0.0
    sig_centered = sig - np.mean(sig)
    autocorr = signal.correlate(sig_centered, sig_centered, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    if autocorr[0] > 0:
        autocorr = autocorr / autocorr[0]
        min_lag, max_lag = int(0.4 * fs), min(int(1.3 * fs), len(autocorr) - 1)
        peaks, _ = signal.find_peaks(autocorr[min_lag:max_lag])
        if len(peaks) > 0:
            return float(np.max(autocorr[min_lag:max_lag][peaks]))
    return 0.0


class BiologicalSignalAnalyzer:
    def __init__(self, fps: float = 30.0, hr_low_hz: float = 0.75, hr_high_hz: float = 3.0):
        """
        Args:
            fps: Video frames per second (default: 30 fps)
            hr_low_hz: Lower bound of normal human heart rate in Hz (0.75 Hz = 45 bpm)
            hr_high_hz: Upper bound of normal human heart rate in Hz (3.0 Hz = 180 bpm)
        """
        self.fps = fps
        self.hr_low_hz = hr_low_hz
        self.hr_high_hz = hr_high_hz

    def bandpass_filter(self, sig: np.ndarray, order: int = 3) -> np.ndarray:
        """
        Applies zero-phase Butterworth bandpass filter in the heart rate band.
        """
        nyquist = 0.5 * self.fps
        low = self.hr_low_hz / nyquist
        high = min(self.hr_high_hz / nyquist, 0.99)
        b, a = signal.butter(order, [low, high], btype='bandpass')
        return signal.filtfilt(b, a, sig)

    def compute_psd_features(self, sig: np.ndarray) -> Dict[str, float]:
        """
        Computes Power Spectral Density (PSD) using Welch's method and extracts
        peak heart rate, SNR, and spectral entropy.
        """
        # Detrend and normalize
        sig_clean = signal.detrend(sig)
        std_val = np.std(sig_clean)
        if std_val > 1e-6:
            sig_clean = sig_clean / std_val

        nperseg = min(len(sig_clean), int(self.fps * 10))
        freqs, psd = signal.welch(sig_clean, fs=self.fps, nperseg=nperseg)

        # Cardiac frequency mask
        hr_mask = (freqs >= self.hr_low_hz) & (freqs <= self.hr_high_hz)
        if not np.any(hr_mask):
            return {
                "peak_hr_bpm": 0.0,
                "snr_db": -20.0,
                "spectral_entropy": 1.0,
                "cardiac_power_ratio": 0.0
            }

        hr_freqs = freqs[hr_mask]
        hr_psd = psd[hr_mask]

        peak_idx = np.argmax(hr_psd)
        peak_freq = hr_freqs[peak_idx]
        peak_hr_bpm = peak_freq * 60.0

        # Signal-to-Noise Ratio (SNR) in cardiac band
        # Cardiac power within ±0.2 Hz of peak
        peak_band_mask = (freqs >= peak_freq - 0.2) & (freqs <= peak_freq + 0.2)
        signal_power = np.sum(psd[peak_band_mask])
        total_power = np.sum(psd)
        noise_power = max(total_power - signal_power, 1e-9)

        snr_db = 10.0 * np.log10((signal_power + 1e-9) / noise_power)
        cardiac_power_ratio = signal_power / max(total_power, 1e-9)

        # Spectral entropy (uniform distribution has max entropy = 1.0)
        norm_psd = hr_psd / (np.sum(hr_psd) + 1e-9)
        spectral_entropy = -np.sum(norm_psd * np.log2(norm_psd + 1e-9))
        max_entropy = np.log2(len(hr_psd)) if len(hr_psd) > 1 else 1.0
        norm_entropy = float(spectral_entropy / max(max_entropy, 1.0))

        return {
            "peak_hr_bpm": float(peak_hr_bpm),
            "snr_db": float(snr_db),
            "spectral_entropy": norm_entropy,
            "cardiac_power_ratio": float(cardiac_power_ratio)
        }

    def analyze(
        self,
        fused_signal: np.ndarray,
        forehead_signal: Optional[np.ndarray] = None,
        cheek_signal: Optional[np.ndarray] = None,
        chin_signal: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Full biological signal analysis with multi-region spatial coherence
        and GAN generative flicker resistance.
        """
        psd_metrics = self.compute_psd_features(fused_signal)

        # Spatial cross-correlation between Forehead and Cheek
        spatial_correlation = 0.0
        if forehead_signal is not None and cheek_signal is not None:
            spatial_correlation = compute_pearson_correlation(forehead_signal, cheek_signal)

        # Chin correlation if available
        chin_corr = 0.0
        if forehead_signal is not None and chin_signal is not None:
            chin_corr = compute_pearson_correlation(forehead_signal, chin_signal)

        # Autocorrelation periodicity
        autocorr_val = compute_autocorrelation_periodicity(fused_signal, fs=self.fps)

        # High-frequency flicker noise (above 4.0 Hz)
        freqs, psd = signal.welch(signal.detrend(fused_signal), fs=self.fps, nperseg=min(len(fused_signal), 128))
        tot_power = np.sum(psd) + 1e-9
        hf_power = np.sum(psd[freqs >= 4.0])
        hf_noise_ratio = float(hf_power / tot_power)

        # Determine authenticity score (0.0 = definite Fake, 1.0 = definite Real)
        # Real videos: High spatial correlation (> 0.25), Good SNR (> -8 dB), Lower entropy (< 0.8), Low HF noise (< 0.08)
        score_snr = np.clip((psd_metrics["snr_db"] + 10.0) / 15.0, 0.0, 1.0)
        score_corr = np.clip((spatial_correlation + 0.1) / 0.8, 0.0, 1.0)
        score_entropy = np.clip(1.0 - psd_metrics["spectral_entropy"], 0.0, 1.0)
        score_periodicity = np.clip(autocorr_val / 0.35, 0.0, 1.0)
        score_hf = np.clip(1.0 - (hf_noise_ratio / 0.15), 0.0, 1.0)

        # Weighted authenticity score incorporating physiological periodicity and GAN flicker
        authenticity_score = float(
            0.30 * score_corr +
            0.25 * score_periodicity +
            0.20 * score_snr +
            0.15 * score_entropy +
            0.10 * score_hf
        )
        is_real = authenticity_score >= 0.50

        return {
            "is_real": is_real,
            "authenticity_score": round(authenticity_score, 4),
            "prediction": "REAL HUMAN" if is_real else "AI-GENERATED / DEEPFAKE",
            "spatial_correlation": round(spatial_correlation, 4),
            "chin_correlation": round(chin_corr, 4) if chin_signal is not None else None,
            "autocorr_periodicity": round(autocorr_val, 4),
            "hf_noise_ratio": round(hf_noise_ratio, 4),
            "heart_rate_bpm": round(psd_metrics["peak_hr_bpm"], 1),
            "snr_db": round(psd_metrics["snr_db"], 2),
            "spectral_entropy": round(psd_metrics["spectral_entropy"], 3),
            "cardiac_power_ratio": round(psd_metrics["cardiac_power_ratio"], 3)
        }
