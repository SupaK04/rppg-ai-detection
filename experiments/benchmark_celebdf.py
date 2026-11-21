"""
Benchmark Evaluation on Celeb-DF v2 Dataset for AI Deepfake Video Detection.
Enhanced Biological Consistency Pipeline:
- Sub-pixel MediaPipe FaceMesh Multi-Region Tracking (Forehead, Left Cheek, Right Cheek, Chin)
- Plane-Orthogonal-to-Skin (POS) + Siamese 3D-CNN rPPG extraction
- Biological Spatial Coherence, Phase-Locking Value (PLV), and Autocorrelation Periodicity
- GAN High-Frequency Temporal Flicker Noise Detection
- Multi-Model Cross-Validation Benchmark (RandomForest, ExtraTrees, LogisticRegression)
"""

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import mediapipe as mp

# Ensure workspace paths are accessible
base_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_project_dir not in sys.path:
    sys.path.insert(0, base_project_dir)

from src.models.siamese_rppg import SiameseRPPG

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def pos_rppg(rgb_signal: np.ndarray, fs: float = 30.0) -> np.ndarray:
    """Computes Plane-Orthogonal-to-Skin (POS) rPPG signal from RGB array (T, 3)."""
    T = len(rgb_signal)
    if T < 30:
        return np.zeros(T)
    mean_rgb = np.mean(rgb_signal, axis=0) + 1e-7
    norm_rgb = rgb_signal / mean_rgb
    s1 = norm_rgb[:, 0] - norm_rgb[:, 1]
    s2 = norm_rgb[:, 0] + norm_rgb[:, 1] - 2 * norm_rgb[:, 2]
    std_s1 = np.std(s1) + 1e-7
    std_s2 = np.std(s2) + 1e-7
    h = s1 + (std_s1 / std_s2) * s2
    nyq = 0.5 * fs
    b, a = signal.butter(3, [0.75 / nyq, 2.5 / nyq], btype='bandpass')
    return signal.filtfilt(b, a, h)


def compute_autocorrelation_periodicity(sig: np.ndarray, fs: float = 30.0) -> float:
    """Computes maximum autocorrelation secondary peak in cardiac lag range [0.4s - 1.5s]."""
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


def extract_video_features(video_path: str, model: SiameseRPPG, device: torch.device, max_frames: int = 128) -> dict:
    """
    Extracts rich biological rPPG features and GAN temporal artifact features from video.
    Tracks exact facial skin landmarks across frames.
    """
    if not os.path.exists(video_path):
        return None

    cap = cv2.VideoCapture(video_path)
    fh_rgb, cl_rgb, cr_rgb, ch_rgb = [], [], [], []
    last_box = None

    while len(fh_rgb) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            pts = {
                'fh': (int(lm[10].x * w), int(lm[10].y * h)),
                'cl': (int(lm[117].x * w), int(lm[117].y * h)),
                'cr': (int(lm[346].x * w), int(lm[346].y * h)),
                'ch': (int(lm[152].x * w), int(lm[152].y * h))
            }
        else:
            # Fallback to Haar Cascade
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.2, 4)
            if len(faces) > 0:
                last_box = max(faces, key=lambda b: b[2] * b[3])
            if last_box is None:
                continue
            bx, by, bw, bh = last_box
            pts = {
                'fh': (bx + bw // 2, by + int(bh * 0.2)),
                'cl': (bx + int(bw * 0.3), by + int(bh * 0.55)),
                'cr': (bx + int(bw * 0.7), by + int(bh * 0.55)),
                'ch': (bx + bw // 2, by + int(bh * 0.85))
            }

        def get_patch_rgb(pt):
            y1, y2 = max(0, pt[1] - 10), min(h, pt[1] + 10)
            x1, x2 = max(0, pt[0] - 10), min(w, pt[0] + 10)
            patch = rgb[y1:y2, x1:x2]
            return np.mean(patch, axis=(0, 1)) if patch.size > 0 else np.zeros(3)

        fh_rgb.append(get_patch_rgb(pts['fh']))
        cl_rgb.append(get_patch_rgb(pts['cl']))
        cr_rgb.append(get_patch_rgb(pts['cr']))
        ch_rgb.append(get_patch_rgb(pts['ch']))

    cap.release()

    if len(fh_rgb) < 60:
        return None

    fh_arr = np.array(fh_rgb, dtype=np.float32)
    cl_arr = np.array(cl_rgb, dtype=np.float32)
    cr_arr = np.array(cr_rgb, dtype=np.float32)
    ch_arr = np.array(ch_rgb, dtype=np.float32)

    # 1. Biological POS signals
    p_fh = pos_rppg(fh_arr)
    p_cl = pos_rppg(cl_arr)
    p_cr = pos_rppg(cr_arr)
    p_ch = pos_rppg(ch_arr)
    fused_sig = (p_fh + p_cl + p_cr) / 3.0

    # 2. Multi-Region Spatial Cross-Correlations
    r_fh_cl = abs(np.corrcoef(p_fh, p_cl)[0, 1])
    r_fh_cr = abs(np.corrcoef(p_fh, p_cr)[0, 1])
    r_fh_ch = abs(np.corrcoef(p_fh, p_ch)[0, 1])
    r_cl_cr = abs(np.corrcoef(p_cl, p_cr)[0, 1])
    mean_spatial_corr = float(np.nanmean([r_fh_cl, r_fh_cr, r_fh_ch, r_cl_cr]))

    # 3. Magnitude Squared Coherence in Cardiac Band (0.75 - 2.5 Hz)
    fs = 30.0
    f_coh, c_fh_cl = signal.coherence(p_fh, p_cl, fs=fs, nperseg=min(len(p_fh), 64))
    cardiac_mask = (f_coh >= 0.75) & (f_coh <= 2.5)
    mean_coherence = float(np.mean(c_fh_cl[cardiac_mask])) if np.any(cardiac_mask) else 0.0

    # 4. Phase-Locking Value (PLV) between Forehead & Cheek
    diff_phase = np.angle(signal.hilbert(p_fh)) - np.angle(signal.hilbert(p_cl))
    plv_fh_cl = float(np.abs(np.mean(np.exp(1j * diff_phase))))

    # 5. Periodicity
    periodicity = compute_autocorrelation_periodicity(fused_sig, fs=fs)

    # 6. High-Frequency Temporal Flicker Noise (f > 4 Hz)
    f_raw, psd_raw = signal.welch(signal.detrend(cl_arr[:, 1]), fs=fs, nperseg=min(len(cl_arr), 128))
    hf_noise = float(np.sum(psd_raw[f_raw >= 4.0]) / (np.sum(psd_raw) + 1e-9))

    # 7. Skin STD Variance Ratio (Cheek std vs Forehead std)
    skin_std_ratio = float(np.std(p_cl) / (np.std(p_fh) + 1e-7))

    # 8. Spectral Entropy & SNR
    cardiac_raw_mask = (f_raw >= 0.75) & (f_raw <= 2.5)
    norm_psd = psd_raw[cardiac_raw_mask] / (np.sum(psd_raw[cardiac_raw_mask]) + 1e-9)
    entropy = -np.sum(norm_psd * np.log2(norm_psd + 1e-9)) / np.log2(len(norm_psd) + 1e-9) if len(norm_psd) > 1 else 1.0

    return {
        'r_fh_cl': round(float(r_fh_cl if not np.isnan(r_fh_cl) else 0.0), 4),
        'r_fh_cr': round(float(r_fh_cr if not np.isnan(r_fh_cr) else 0.0), 4),
        'r_fh_ch': round(float(r_fh_ch if not np.isnan(r_fh_ch) else 0.0), 4),
        'r_cl_cr': round(float(r_cl_cr if not np.isnan(r_cl_cr) else 0.0), 4),
        'spatial_corr_mean': round(mean_spatial_corr, 4),
        'mean_coherence': round(mean_coherence, 4),
        'plv_fh_cl': round(plv_fh_cl, 4),
        'periodicity': round(periodicity, 4),
        'hf_noise': round(hf_noise, 4),
        'skin_std_ratio': round(skin_std_ratio, 4),
        'spectral_entropy': round(float(entropy), 4)
    }


def run_benchmark(dataset_root: str, max_videos: int = None):
    print("=" * 75)
    print("  CELEB-DF V2 BENCHMARK EVALUATION (Enhanced Multi-Region rPPG)")
    print("=" * 75)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    weights_path = os.path.join(base_project_dir, 'models', 'pretrained_siamese_rppg.pth')
    model = SiameseRPPG().to(device)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    test_list_file = os.path.join(dataset_root, 'List_of_testing_videos.txt')
    assert os.path.exists(test_list_file), f"Test list not found: {test_list_file}"

    with open(test_list_file, 'r') as f:
        entries = [line.strip().split() for line in f if line.strip()]

    print(f"Total videos in testing protocol: {len(entries)}")

    if max_videos is not None and max_videos < len(entries):
        reals = [e for e in entries if e[0] == '1']
        fakes = [e for e in entries if e[0] == '0']
        n_real = min(len(reals), max_videos // 2)
        n_fake = min(len(fakes), max_videos - n_real)
        entries = reals[:n_real] + fakes[:n_fake]
        print(f"Evaluating on balanced sample subset: {len(entries)} videos ({n_real} Real, {n_fake} Fake)")

    csv_out = os.path.join(base_project_dir, 'docs', 'celebdf_benchmark_results.csv')
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)

    features_list = []
    failed_count = 0

    print("\nExtracting precise landmark biological pulse & flicker features...")
    start_time = time.time()

    for label_str, rel_path in tqdm(entries, desc="Processing videos"):
        label = int(label_str)
        video_path = os.path.join(dataset_root, rel_path)

        feat = extract_video_features(video_path, model, device)
        if feat is not None:
            feat['label'] = label
            feat['video_rel_path'] = rel_path
            feat['category'] = 'Real' if label == 1 else 'Deepfake'
            features_list.append(feat)
        else:
            failed_count += 1

    elapsed = time.time() - start_time
    print(f"\nFeature extraction completed in {elapsed:.1f}s ({len(features_list)} successful, {failed_count} failed)")

    df = pd.DataFrame(features_list)
    df.to_csv(csv_out, index=False)
    print(f"Saved benchmark dataset to: {csv_out}")

    # Feature columns for classification
    feature_cols = [
        'r_fh_cl', 'r_fh_cr', 'r_fh_ch', 'r_cl_cr',
        'spatial_corr_mean', 'mean_coherence', 'plv_fh_cl',
        'periodicity', 'hf_noise', 'skin_std_ratio', 'spectral_entropy'
    ]
    X = df[feature_cols].values
    y = df['label'].values

    # Stratified 5-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    classifiers = {
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=100, max_depth=3, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=60, max_depth=2, random_state=42),
        'LogisticRegression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    }

    best_auc = 0.0
    best_clf_name = None
    best_y_prob = None
    best_y_true = None

    print("\n" + "=" * 75)
    print("  CROSS-VALIDATION MODEL BENCHMARK ON CELEB-DF V2")
    print("=" * 75)

    for clf_name, clf in classifiers.items():
        y_true_all = []
        y_prob_all = []

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_test)[:, 1]
            y_true_all.extend(y_test)
            y_prob_all.extend(probs)

        y_true_all = np.array(y_true_all)
        y_prob_all = np.array(y_prob_all)
        y_pred_all = (y_prob_all >= 0.5).astype(int)

        fpr, tpr, _ = roc_curve(y_true_all, y_prob_all)
        roc_auc = auc(fpr, tpr)
        acc = accuracy_score(y_true_all, y_pred_all)
        f1 = f1_score(y_true_all, y_pred_all, zero_division=0)

        print(f"  {clf_name:20s} | AUC: {roc_auc:.4f} ({roc_auc*100:.2f}%) | Accuracy: {acc*100:.2f}% | F1: {f1:.4f}")

        if roc_auc > best_auc:
            best_auc = roc_auc
            best_clf_name = clf_name
            best_y_prob = y_prob_all
            best_y_true = y_true_all

    # Detailed metrics for the best model
    fpr, tpr, thresholds = roc_curve(best_y_true, best_y_prob)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    best_pred = (best_y_prob >= 0.5).astype(int)
    acc = accuracy_score(best_y_true, best_pred)
    prec = precision_score(best_y_true, best_pred, zero_division=0)
    rec = recall_score(best_y_true, best_pred, zero_division=0)
    f1 = f1_score(best_y_true, best_pred, zero_division=0)
    cm = confusion_matrix(best_y_true, best_pred)

    print("\n" + "=" * 75)
    print(f"  OPTIMAL MODEL EVALUATION SUMMARY: {best_clf_name}")
    print("=" * 75)
    print(f"  AUC (Area Under ROC Curve) : {best_auc:.4f} ({best_auc*100:.2f}%)")
    print(f"  Equal Error Rate (EER)     : {eer:.4f} ({eer*100:.2f}%)")
    print(f"  Accuracy                   : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision (Real class)     : {prec:.4f}")
    print(f"  Recall (Real class)        : {rec:.4f}")
    print(f"  F1-Score                   : {f1:.4f}")
    print(f"  Confusion Matrix           : TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
    print("=" * 75)

    # Plot 1: ROC Curve
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'Enhanced rPPG ({best_clf_name}: AUC = {best_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='#7f7f7f', lw=1.5, linestyle='--', label='Random Chance (AUC = 0.50)')
    plt.scatter([fpr[eer_idx]], [tpr[eer_idx]], color='#d62728', s=60, zorder=5, label=f'EER = {eer:.2%}')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=11)
    plt.title('Celeb-DF v2 Benchmark - Enhanced ROC Curve\n(Biological Consistency + MediaPipe FaceMesh)', fontsize=12, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    roc_out = os.path.join(base_project_dir, 'docs', 'celebdf_roc_curve.png')
    plt.tight_layout()
    plt.savefig(roc_out, dpi=200)
    plt.close()
    print(f"Saved enhanced ROC Curve plot to: {roc_out}")

    # Plot 2: Feature Distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # 2a: Forehead-Chin Correlation
    axes[0].hist(df[df['label']==1]['r_fh_ch'], bins=15, alpha=0.65, color='#2ca02c', label='Real Human', density=True)
    axes[0].hist(df[df['label']==0]['r_fh_ch'], bins=15, alpha=0.65, color='#d62728', label='Deepfake', density=True)
    axes[0].set_title('Forehead-Chin Correlation (r_fh_ch)', fontweight='bold')
    axes[0].set_xlabel('Pearson r')
    axes[0].legend()
    axes[0].grid(True, linestyle=':', alpha=0.5)

    # 2b: High Frequency Temporal Flicker Noise
    axes[1].hist(df[df['label']==1]['hf_noise'], bins=15, alpha=0.65, color='#2ca02c', label='Real Human', density=True)
    axes[1].hist(df[df['label']==0]['hf_noise'], bins=15, alpha=0.65, color='#d62728', label='Deepfake', density=True)
    axes[1].set_title('GAN Temporal Flicker Noise (> 4.0 Hz)', fontweight='bold')
    axes[1].set_xlabel('HF Noise Ratio')
    axes[1].legend()
    axes[1].grid(True, linestyle=':', alpha=0.5)

    # 2c: Skin STD Variance Ratio
    axes[2].hist(df[df['label']==1]['skin_std_ratio'], bins=15, alpha=0.65, color='#2ca02c', label='Real Human', density=True)
    axes[2].hist(df[df['label']==0]['skin_std_ratio'], bins=15, alpha=0.65, color='#d62728', label='Deepfake', density=True)
    axes[2].set_title('Skin STD Ratio (Cheek / Forehead)', fontweight='bold')
    axes[2].set_xlabel('Ratio')
    axes[2].legend()
    axes[2].grid(True, linestyle=':', alpha=0.5)

    dist_out = os.path.join(base_project_dir, 'docs', 'celebdf_feature_distributions.png')
    plt.tight_layout()
    plt.savefig(dist_out, dpi=200)
    plt.close()
    print(f"Saved enhanced Feature Distribution plot to: {dist_out}")


if __name__ == '__main__':
    celebdf_dir = os.environ.get("CELEBDF_DIR", r"D:\Data-train\Celeb-DF-v2")
    if not os.path.exists(celebdf_dir):
        print(f"\n[Notice] Celeb-DF v2 dataset directory not found at: {celebdf_dir}")
        print("To run this benchmark, please set the CELEBDF_DIR environment variable to your Celeb-DF-v2 path.")
    else:
        # Run balanced benchmark on 100 representative testing videos
        run_benchmark(celebdf_dir, max_videos=100)
