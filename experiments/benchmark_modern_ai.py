"""
Benchmark Evaluation on Modern Generative AI & Deepfake Videos (2024-2026).
Tests Siamese-rPPG on modern generative models:
1. Alibaba HumanAIGC EMO (Audio-driven Diffusion Avatars)
2. KwaiVGI / Kling AI LivePortrait (Portrait Animation & Motion Retargeting)
3. OpenTalker SadTalker (3DMM Latent Audio-driven Talking Head)
4. Commercial AI FaceSwap (MagicHour / FaceSwapVideo) vs Real References
"""

import sys
import os
import time
import cv2
import numpy as np
import pandas as pd
import torch
from scipy import signal
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.models.siamese_rppg import SiameseRPPG


def compute_autocorrelation_peak(sig: np.ndarray, fs: float = 30.0) -> float:
    if len(sig) < int(fs * 2):
        return 0.0
    sig_centered = sig - np.mean(sig)
    autocorr = signal.correlate(sig_centered, sig_centered, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    if autocorr[0] > 0:
        autocorr = autocorr / autocorr[0]
    else:
        return 0.0

    min_lag = int(0.4 * fs)
    max_lag = min(int(1.5 * fs), len(autocorr) - 1)
    if min_lag < max_lag:
        peaks, _ = signal.find_peaks(autocorr[min_lag:max_lag])
        if len(peaks) > 0:
            return float(np.max(autocorr[min_lag:max_lag][peaks]))
    return 0.0


def extract_features_from_video(video_path: str, model: SiameseRPPG, device: torch.device, max_frames: int = 128):
    if not os.path.exists(video_path):
        return None

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps) or fps > 120:
        fps = 30.0

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    forehead_list = []
    cheek_list = []
    last_box = None

    while len(forehead_list) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        fh, fw = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(60, 60))
        if len(faces) > 0:
            last_box = max(faces, key=lambda b: b[2] * b[3])

        if last_box is not None:
            bx, by, bw, bh = last_box
            fy1 = max(0, by + int(bh * 0.08))
            fy2 = min(fh, by + int(bh * 0.32))
            fx1 = max(0, bx + int(bw * 0.20))
            fx2 = min(fw, bx + int(bw * 0.80))

            cy1 = max(0, by + int(bh * 0.50))
            cy2 = min(fh, by + int(bh * 0.76))
            cx1 = max(0, bx + int(bw * 0.15))
            cx2 = min(fw, bx + int(bw * 0.85))

            f_crop = frame[fy1:fy2, fx1:fx2]
            c_crop = frame[cy1:cy2, cx1:cx2]

            if f_crop.size > 0 and c_crop.size > 0:
                f_resized = cv2.cvtColor(cv2.resize(f_crop, (140, 40)), cv2.COLOR_BGR2RGB)
                c_resized = cv2.cvtColor(cv2.resize(c_crop, (140, 40)), cv2.COLOR_BGR2RGB)
                forehead_list.append(f_resized)
                cheek_list.append(c_resized)

    cap.release()

    if len(forehead_list) < 64:
        return None

    # Truncate to power-of-2 or 128 frames
    n_frames = min(len(forehead_list), max_frames)
    f_arr = np.array(forehead_list[:n_frames], dtype=np.float32)
    c_arr = np.array(cheek_list[:n_frames], dtype=np.float32)

    # DiffNormalized temporal signals
    diff_f = (f_arr[1:] - f_arr[:-1]) / (f_arr[1:] + f_arr[:-1] + 1e-7)
    diff_f = diff_f / (np.std(diff_f) + 1e-7)
    diff_f = np.append(diff_f, np.zeros_like(diff_f[:1]), axis=0)
    diff_f[np.isnan(diff_f)] = 0

    diff_c = (c_arr[1:] - c_arr[:-1]) / (c_arr[1:] + c_arr[:-1] + 1e-7)
    diff_c = diff_c / (np.std(diff_c) + 1e-7)
    diff_c = np.append(diff_c, np.zeros_like(diff_c[:1]), axis=0)
    diff_c[np.isnan(diff_c)] = 0

    t_f = torch.from_numpy(diff_f).permute(3, 0, 1, 2).unsqueeze(0).to(device)
    t_c = torch.from_numpy(diff_c).permute(3, 0, 1, 2).unsqueeze(0).to(device)

    with torch.no_grad():
        fh_sig = model.shared_branch(t_f).squeeze().cpu().numpy()
        ck_sig = model.shared_branch(t_c).squeeze().cpu().numpy()
        f_sig = fh_sig + ck_sig

    # 1. Bandpass filter (0.75 - 3.0 Hz)
    nyquist = 0.5 * fps
    b, a = signal.butter(3, [0.75 / nyquist, 3.0 / nyquist], btype='bandpass')
    filtered_sig = signal.filtfilt(b, a, f_sig)
    fh_filt = signal.filtfilt(b, a, fh_sig)
    ck_filt = signal.filtfilt(b, a, ck_sig)

    # 2. Spatial Correlation
    r = np.corrcoef(fh_filt, ck_filt)[0, 1]
    spatial_corr = float(r) if not np.isnan(r) else 0.0

    # 3. Welch PSD
    freqs, psd = signal.welch(signal.detrend(filtered_sig), fs=fps, nperseg=min(len(filtered_sig), 128), nfft=512)
    hr_mask = (freqs >= 0.75) & (freqs <= 3.0)

    if np.any(hr_mask):
        hr_psd = psd[hr_mask]
        peak_idx = np.argmax(hr_psd)
        peak_freq = freqs[hr_mask][peak_idx]
        peak_hr_bpm = peak_freq * 60.0

        peak_band = (freqs >= peak_freq - 0.2) & (freqs <= peak_freq + 0.2)
        sig_power = np.sum(psd[peak_band])
        tot_power = np.sum(psd)
        snr_db = 10.0 * np.log10((sig_power + 1e-9) / max(tot_power - sig_power, 1e-9))
        cpr = sig_power / max(tot_power, 1e-9)

        norm_psd = hr_psd / (np.sum(hr_psd) + 1e-9)
        entropy = -np.sum(norm_psd * np.log2(norm_psd + 1e-9)) / np.log2(len(hr_psd))
    else:
        peak_hr_bpm = 0.0
        snr_db = -20.0
        cpr = 0.0
        entropy = 1.0

    autocorr_val = compute_autocorrelation_peak(filtered_sig, fs=fps)

    # Scoring
    score_corr = np.clip((spatial_corr + 0.05) / 0.55, 0.0, 1.0)
    score_auto = np.clip(autocorr_val / 0.35, 0.0, 1.0)
    score_snr = np.clip((snr_db + 8.0) / 11.0, 0.0, 1.0)
    score_hr = 1.0 if (50.0 <= peak_hr_bpm <= 125.0) else 0.4
    score_entropy = np.clip((0.94 - entropy) / 0.25, 0.0, 1.0)

    liveness_score = float(
        0.30 * score_corr +
        0.30 * score_auto +
        0.20 * score_snr +
        0.10 * score_hr +
        0.10 * score_entropy
    )

    is_real = liveness_score >= 0.45

    return {
        "heart_rate_bpm": round(float(peak_hr_bpm), 1),
        "spatial_corr": round(spatial_corr, 3),
        "autocorr_peak": round(autocorr_val, 3),
        "snr_db": round(float(snr_db), 1),
        "spectral_entropy": round(float(entropy), 2),
        "liveness_score": round(liveness_score, 4),
        "predicted_real": is_real,
        "prediction_str": "REAL HUMAN" if is_real else "AI-GENERATED / DEEPFAKE"
    }


def main():
    print("=" * 75)
    print("  MODERN GENERATIVE AI & DEEPFAKE BENCHMARK EVALUATION (2024-2026)")
    print("  Siamese-rPPG Biological Signal Verification")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load Model
    model = SiameseRPPG().to(device)
    weights_path = os.path.join(BASE_DIR, "models", "pretrained_siamese_rppg.pth")
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded Siamese-rPPG weights from: {weights_path}")

    dataset_dir = os.environ.get("MODERN_AI_DIR", r"D:\Data-train\Modern-AI-Deepfakes")
    if not os.path.exists(dataset_dir):
        print(f"\n[Notice] Modern AI Deepfakes dataset directory not found at: {dataset_dir}")
        print("To run this benchmark, please set the MODERN_AI_DIR environment variable to your dataset path.")
        return

    metadata = [
        # 1. Diffusion Video Avatars (Audio-driven, Alibaba HumanAIGC EMO)
        {"file": "EMO_avatar_main.mp4", "gen_model": "EMO (Alibaba)", "type": "Diffusion Avatar", "true_label": 0},
        {"file": "EMO_song_melody.mp4", "gen_model": "EMO (Alibaba)", "type": "Diffusion Avatar", "true_label": 0},
        {"file": "EMO_talk_sora.mp4", "gen_model": "EMO (Alibaba)", "type": "Diffusion Avatar", "true_label": 0},
        
        # 2. 3DMM Latent Audio-driven Talking Head (OpenTalker SadTalker)
        {"file": "SadTalker_AOC_000.mp4", "gen_model": "SadTalker (OpenTalker)", "type": "3DMM Talking Head", "true_label": 0},
        {"file": "SadTalker_KatieHill_000.mp4", "gen_model": "SadTalker (OpenTalker)", "type": "3DMM Talking Head", "true_label": 0},
        
        # 3. Commercial AI FaceSwap (MagicHour / FaceSwapVideo.ai)
        {"file": "Modern_FaceSwap_1_fake.mp4", "gen_model": "Commercial FaceSwap AI", "type": "AI FaceSwap", "true_label": 0},
        {"file": "Modern_FaceSwap_2_fake.mp4", "gen_model": "Commercial FaceSwap AI", "type": "AI FaceSwap", "true_label": 0},
        {"file": "Modern_FaceSwap_3_fake.mp4", "gen_model": "Commercial FaceSwap AI", "type": "AI FaceSwap", "true_label": 0},

        # 4. Real Human Driving Videos (KwaiVGI LivePortrait Driving Human Subjects)
        {"file": "LivePortrait_driving_d0.mp4", "gen_model": "Real Human Driving Subject", "type": "Real Human Video", "true_label": 1},
        {"file": "LivePortrait_driving_d6.mp4", "gen_model": "Real Human Driving Subject", "type": "Real Human Video", "true_label": 1},
        {"file": "LivePortrait_driving_d14.mp4", "gen_model": "Real Human Driving Subject", "type": "Real Human Video", "true_label": 1},

        # 5. Real Camera Reference Videos (Source unmanipulated videos)
        {"file": "Modern_FaceSwap_1_real.mp4", "gen_model": "Real Camera Video", "type": "Real Human Video", "true_label": 1},
        {"file": "Modern_FaceSwap_2_real.mp4", "gen_model": "Real Camera Video", "type": "Real Human Video", "true_label": 1},
        {"file": "Modern_FaceSwap_3_real.mp4", "gen_model": "Real Camera Video", "type": "Real Human Video", "true_label": 1},
    ]

    results = []

    print("\nProcessing modern AI videos through Siamese-rPPG pipeline...")
    for item in metadata:
        vpath = os.path.join(dataset_dir, item["file"])
        if not os.path.exists(vpath):
            print(f"  [Skip] Missing: {item['file']}")
            continue

        res = extract_features_from_video(vpath, model, device)
        if res is None:
            print(f"  [Warning] Face could not be tracked in: {item['file']}")
            continue

        is_correct = (res["predicted_real"] == (item["true_label"] == 1))
        record = {
            "Video File": item["file"],
            "Model / Generator": item["gen_model"],
            "Category": item["type"],
            "Ground Truth": "Real" if item["true_label"] == 1 else "Deepfake / AI",
            "Prediction": res["prediction_str"],
            "Liveness Score (%)": round(res["liveness_score"] * 100, 1),
            "Heart Rate (BPM)": res["heart_rate_bpm"],
            "Spatial Corr (r)": res["spatial_corr"],
            "Periodicity Peak": res["autocorr_peak"],
            "SNR (dB)": res["snr_db"],
            "Spectral Entropy": res["spectral_entropy"],
            "Correct Detection": "✓" if is_correct else "✗"
        }
        results.append(record)
        print(f"  [{record['Correct Detection']}] {item['file']} ({item['gen_model']}): {record['Prediction']} (Liveness: {record['Liveness Score (%)']}%, r={res['spatial_corr']}, AutoCorr={res['autocorr_peak']})")

    df = pd.DataFrame(results)

    # Save CSV
    out_csv = os.path.join(BASE_DIR, "docs", "modern_ai_benchmark_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved benchmark results to:\n  {out_csv}")

    # Accuracy Calculation
    correct_count = sum(1 for r in results if r["Correct Detection"] == "✓")
    total_count = len(results)
    acc = (correct_count / total_count) * 100 if total_count > 0 else 0
    print(f"\nOverall Modern AI Detection Accuracy: {correct_count}/{total_count} ({acc:.1f}%)")

    # Plot Comparison
    plt.figure(figsize=(10, 6))
    categories = df["Category"].unique()
    colors = {"Diffusion Avatar": "#e74c3c", "Portrait Animation": "#e67e22", "3DMM Talking Head": "#9b59b6", "AI FaceSwap": "#c0392b", "Real Reference": "#27ae60"}

    for cat in categories:
        sub = df[df["Category"] == cat]
        plt.scatter(
            sub["Spatial Corr (r)"],
            sub["Periodicity Peak"],
            label=f"{cat} ({len(sub)})",
            color=colors.get(cat, "#34495e"),
            s=120,
            alpha=0.85,
            edgecolors='black',
            linewidth=1.2
        )

    plt.axvline(0.20, color='gray', linestyle='--', alpha=0.6, label='Spatial Corr Threshold (r=0.20)')
    plt.axhline(0.20, color='gray', linestyle=':', alpha=0.6, label='Periodicity Threshold (0.20)')
    plt.xlabel("Spatial Cross-Correlation between Forehead & Cheek (Pearson r)", fontsize=11)
    plt.ylabel("Temporal Periodicity Peak (Autocorrelation)", fontsize=11)
    plt.title("Biological Pulse Consistency: Real Humans vs Modern Generative AI Videos (2024-2026)", fontsize=12, fontweight='bold')
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.5)

    plot_out = os.path.join(BASE_DIR, "docs", "modern_ai_benchmark_comparison.png")
    plt.tight_layout()
    plt.savefig(plot_out, dpi=200)
    plt.close()
    print(f"Saved visualization plot to:\n  {plot_out}")


if __name__ == "__main__":
    main()
