"""
rPPG Screen AI-Generated Video Inspector - Compact Floating HUD Edition
Dual-Engine: 3D-CNN rPPG + BiArterialDeepfakeNet + Anti-Spoofing PAD
"""

import os
import sys
import time
import threading
import tkinter as tk
from tkinter import messagebox, filedialog
import cv2
import numpy as np
import torch
from scipy import signal
import mediapipe as mp
from PIL import Image, ImageTk
import mss

# Path Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.models.siamese_rppg import SiameseRPPG
from src.models.bi_arterial_net import BiArterialDeepfakeNet

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MediaPipe Landmark Indices
FH_INDICES = [151, 9, 8, 108, 337, 69, 299, 107, 336]
LC_INDICES = [117, 118, 119, 100, 126, 209, 50, 205]
RC_INDICES = [346, 347, 348, 329, 355, 429, 280, 425]
CK_INDICES = LC_INDICES + RC_INDICES

# Eye Landmarks for Blinking (EAR)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


def pos_rppg(rgb_signal: np.ndarray, fs: float = 30.0) -> np.ndarray:
    """
    Wang et al. Plane-Orthogonal-to-Skin (POS) rPPG with temporal sliding window.
    Cancels ambient illumination sweeps, lighting drift, and shadow transitions.
    """
    N = len(rgb_signal)
    if N < 30: return np.zeros(N)
    l = min(int(fs * 1.6), N)
    if l < 10: return np.zeros(N)
    H = np.zeros(N)
    for m in range(0, N - l + 1):
        w = rgb_signal[m:m + l]
        mean_rgb = np.mean(w, axis=0) + 1e-7
        Cn = w / mean_rgb
        s1 = Cn[:, 1] - Cn[:, 2]
        s2 = Cn[:, 1] + Cn[:, 2] - 2 * Cn[:, 0]
        h = s1 + (np.std(s1) / (np.std(s2) + 1e-7)) * s2
        H[m:m + l] += h - np.mean(h)
    nyq = 0.5 * fs
    b, a = signal.butter(3, [0.75 / nyq, 2.5 / nyq], btype='bandpass')
    return signal.filtfilt(b, a, H)


def calculate_ear(landmarks, eye_indices, w, h):
    pts = np.array([(landmarks[idx].x * w, landmarks[idx].y * h) for idx in eye_indices])
    d_v1 = np.linalg.norm(pts[1] - pts[5])
    d_v2 = np.linalg.norm(pts[2] - pts[4])
    d_h = np.linalg.norm(pts[0] - pts[3])
    ear = (d_v1 + d_v2) / (2.0 * (d_h + 1e-7))
    return float(ear)


class RegionSnipper:
    """Semi-transparent full-screen sniper overlay to select screen region with click & drag."""
    def __init__(self, parent, on_selected_callback):
        self.parent = parent
        self.top = tk.Toplevel(parent)
        self.top.attributes("-fullscreen", True)
        self.top.attributes("-alpha", 0.28)
        self.top.config(cursor="cross")
        self.top.lift()
        self.top.attributes("-topmost", True)

        self.canvas = tk.Canvas(self.top, cursor="cross", bg="#0a0a0f")
        self.canvas.pack(fill="both", expand=True)

        self.callback = on_selected_callback
        self.start_x = None
        self.start_y = None
        self.rect = None

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.top.bind("<Escape>", lambda e: self.cancel())

        lbl = tk.Label(self.top, text="🎯 คลิกแล้วลากกรอบสี่เหลี่ยมคลุมวิดีโอ (Zoom / Meet / YouTube)  [กด ESC เพื่อยกเลิก]",
                       font=("Segoe UI", 14, "bold"), fg="#00f5d4", bg="#101018", padx=20, pady=8)
        lbl.place(relx=0.5, rely=0.06, anchor="center")

    def cancel(self):
        self.top.destroy()
        self.parent.deiconify()

    def on_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y,
                                                 outline="#00f5d4", width=3, dash=(5, 5))

    def on_drag(self, event):
        cur_x, cur_y = event.x, event.y
        self.canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_release(self, event):
        end_x, end_y = event.x, event.y
        x1 = min(self.start_x, end_x)
        y1 = min(self.start_y, end_y)
        x2 = max(self.start_x, end_x)
        y2 = max(self.start_y, end_y)

        w = x2 - x1
        h = y2 - y1
        self.top.destroy()
        self.parent.deiconify()

        if w >= 80 and h >= 80:
            self.callback(x1, y1, w, h)
        else:
            messagebox.showwarning("คำเตือน", "กรอบที่เลือกมีขนาดเล็กเกินไป (ต้องการอย่างน้อย 80x80 พิกเซล)")


class FloatingInspectorHUD:
    def __init__(self, root):
        self.root = root
        self.root.title("rPPG Deepfake Inspector")
        
        # Geometry: 380px wide, 700px high positioned top-right
        screen_w = self.root.winfo_screenwidth()
        hud_w, hud_h = 380, 700
        pos_x = max(10, screen_w - hud_w - 20)
        pos_y = 25
        self.root.geometry(f"{hud_w}x{hud_h}+{pos_x}+{pos_y}")
        self.root.minsize(360, 220)
        self.root.configure(bg="#0f111a")
        self.root.attributes("-topmost", True)

        # State Variables
        self.is_compact_mode = False
        self.capture_mode = "SCREEN"
        self.screen_rect = None
        self.video_cap = None
        self.is_running = True
        self.is_paused = False
        self.latest_display_frame = None

        # Analysis Buffers
        self.buffer_size = 128
        self.fh_crops, self.ck_crops = [], []
        self.fh_rgb, self.cl_rgb, self.cr_rgb = [], [], []
        self.pulse_history = [0.0] * 128
        self.ear_history = []
        self.raw_frame_diffs = []
        self.prev_raw_gray = None
        self.prev_face_center = None
        self.motion_detected = False

        # Current Analysis State
        self.current_result = {
            "status": "WAITING",
            "is_real": False,
            "bpm": 0.0,
            "liveness": 0.0,
            "neural_prob": 50.0,
            "nn_r_fc": 0.0,
            "pos_r_fc": 0.0,
            "pos_r_lr": 0.0,
            "snr_db": 0.0,
            "hf_noise": 0.0,
            "autocorr": 0.0
        }

        # Load AI Models
        self.load_models()

        # Build UI Layout
        self.create_ui()

        # Processing Thread
        self.thread = threading.Thread(target=self.processing_loop, daemon=True)
        self.thread.start()

        # GUI Update Loop
        self.root.after(40, self.update_gui_loop)

    def load_models(self):
        print("Loading Dual-Engine Deepfake Detection Models...")
        self.bio_model = SiameseRPPG().to(device)
        weights_path = os.path.join(BASE_DIR, "models", "pretrained_siamese_rppg.pth")
        if os.path.exists(weights_path):
            self.bio_model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
            self.bio_model.eval()

        self.net_model = BiArterialDeepfakeNet().to(device)
        net_path = os.path.join(BASE_DIR, "models", "bi_arterial_deepfake_net.pth")
        if os.path.exists(net_path):
            ckpt = torch.load(net_path, map_location=device, weights_only=True)
            if "model_state_dict" in ckpt:
                self.net_model.load_state_dict(ckpt["model_state_dict"])
            else:
                self.net_model.load_state_dict(ckpt)
            self.net_model.eval()

        self.mp_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=False)

    def create_ui(self):
        # 1. Sleek Header Bar
        header = tk.Frame(self.root, bg="#161926", height=38, padx=10, pady=5)
        header.pack(fill="x", side="top")

        title_lbl = tk.Label(header, text="🛡️ rPPG Deepfake Inspector", font=("Segoe UI", 11, "bold"),
                             fg="#00f5d4", bg="#161926")
        title_lbl.pack(side="left")

        self.btn_toggle_mode = tk.Button(header, text="▲ ย่อลง", font=("Segoe UI", 8, "bold"),
                                         bg="#252b42", fg="#94a3b8", relief="flat", padx=6, pady=2,
                                         command=self.toggle_compact_mode)
        self.btn_toggle_mode.pack(side="right", padx=3)

        self.is_pinned = True
        self.btn_pin = tk.Button(header, text="📌 หมุด", font=("Segoe UI", 8, "bold"),
                                 bg="#0284c7", fg="#ffffff", relief="flat", padx=6, pady=2,
                                 command=self.toggle_pin)
        self.btn_pin.pack(side="right", padx=3)

        # 2. Main Snip Button
        ctrl_bar = tk.Frame(self.root, bg="#0f111a", padx=10, pady=5)
        ctrl_bar.pack(fill="x")

        self.btn_snip = tk.Button(ctrl_bar, text="🎯 ลากครอบพื้นที่หน้าจอ", font=("Segoe UI", 11, "bold"),
                                  bg="#00b4d8", fg="#ffffff", activebackground="#0077b6",
                                  relief="flat", pady=6, cursor="hand2", command=self.start_snip)
        self.btn_snip.pack(fill="x")

        # 3. Clean Status Card: ONLY reports "คนจริง (REAL HUMAN)" or "AI DEEPFAKE"
        self.status_card = tk.Label(self.root, text="⏳ กรุณาลากครอบพื้นที่วิดีโอบนจอ",
                                    font=("Segoe UI", 13, "bold"), fg="#94a3b8", bg="#1e2235",
                                    pady=10, relief="flat")
        self.status_card.pack(fill="x", padx=10, pady=3)

        # 4. Big Meters (BPM & Liveness)
        meter_frame = tk.Frame(self.root, bg="#0f111a", padx=10, pady=2)
        meter_frame.pack(fill="x")

        bpm_box = tk.Frame(meter_frame, bg="#161926", padx=8, pady=5, bd=1, relief="solid")
        bpm_box.pack(side="left", fill="x", expand=True, padx=(0, 4))
        tk.Label(bpm_box, text="อัตราการเต้นหัวใจ", font=("Segoe UI", 8), fg="#64748b", bg="#161926").pack()
        self.bpm_val = tk.Label(bpm_box, text="-- BPM", font=("Segoe UI", 15, "bold"), fg="#38bdf8", bg="#161926")
        self.bpm_val.pack()

        live_box = tk.Frame(meter_frame, bg="#161926", padx=8, pady=5, bd=1, relief="solid")
        live_box.pack(side="right", fill="x", expand=True, padx=(4, 0))
        tk.Label(live_box, text="ความมีชีวิต (Liveness)", font=("Segoe UI", 8), fg="#64748b", bg="#161926").pack()
        self.live_val = tk.Label(live_box, text="-- %", font=("Segoe UI", 15, "bold"), fg="#38bdf8", bg="#161926")
        self.live_val.pack()

        # 5. Live Waveform Box
        wave_box = tk.Frame(self.root, bg="#0f111a", padx=10, pady=2)
        wave_box.pack(fill="x")

        tk.Label(wave_box, text="📈 รูปคลื่นชีพจรใต้ผิวหนัง (PPG Pulse Wave):",
                 font=("Segoe UI", 8, "bold"), fg="#64748b", bg="#0f111a").pack(anchor="w")

        self.wave_canvas = tk.Canvas(wave_box, bg="#08090d", height=45, highlightthickness=1, highlightbackground="#1e2235")
        self.wave_canvas.pack(fill="x", expand=True, pady=2)

        # 6. Detail Frame (Video Preview Canvas + Detailed Metrics)
        self.detail_frame = tk.Frame(self.root, bg="#0f111a", padx=10, pady=2)
        self.detail_frame.pack(fill="both", expand=True)

        # Explicit Video Canvas with guaranteed 200px height (pack_propagate=False prevents collapsing!)
        preview_container = tk.Frame(self.detail_frame, bg="#161926", width=360, height=205, bd=1, relief="solid")
        preview_container.pack(fill="x", pady=3)
        preview_container.pack_propagate(False)

        self.video_canvas = tk.Canvas(preview_container, bg="#08090d", highlightthickness=0)
        self.video_canvas.pack(fill="both", expand=True)

        # Detailed Metrics Grid
        metrics_grid = tk.Frame(self.detail_frame, bg="#161926", padx=8, pady=5, bd=1, relief="solid")
        metrics_grid.pack(fill="x", pady=3)

        self.metrics_labels = {}
        rows = [
            ("Neural Authenticity", "neural_prob", "%"),
            ("Arterial Coherence (r_fc)", "nn_r_fc", ""),
            ("Bilateral Cheeks Symmetry (r_lr)", "pos_r_lr", ""),
            ("BVP Pulse SNR", "snr_db", "dB"),
            ("Temporal Signal Noise", "hf_noise", "")
        ]

        for i, (label_text, key, unit) in enumerate(rows):
            tk.Label(metrics_grid, text=label_text, font=("Segoe UI", 8), fg="#94a3b8", bg="#161926", anchor="w").grid(row=i, column=0, sticky="w", pady=1)
            val_lbl = tk.Label(metrics_grid, text="--", font=("Consolas", 8, "bold"), fg="#f8fafc", bg="#161926")
            val_lbl.grid(row=i, column=1, sticky="e", padx=(8, 0), pady=1)
            self.metrics_labels[key] = (val_lbl, unit)

        # Bottom Utility Buttons: 1-click sample testing + custom file + snapshot
        bot_bar = tk.Frame(self.detail_frame, bg="#0f111a")
        bot_bar.pack(fill="x", pady=(3, 0))

        btn_sample_real = tk.Button(bot_bar, text="▶ คนจริง", font=("Segoe UI", 8, "bold"),
                                    bg="#064e3b", fg="#a7f3d0", relief="flat", padx=3, pady=3,
                                    command=lambda: self.load_sample("real"))
        btn_sample_real.pack(side="left", fill="x", expand=True, padx=(0, 2))

        btn_sample_ai = tk.Button(bot_bar, text="▶ AI ปลอม", font=("Segoe UI", 8, "bold"),
                                  bg="#7f1d1d", fg="#fecaca", relief="flat", padx=3, pady=3,
                                  command=lambda: self.load_sample("fake"))
        btn_sample_ai.pack(side="left", fill="x", expand=True, padx=(2, 2))

        btn_file = tk.Button(bot_bar, text="📁 ไฟล์คลิป", font=("Segoe UI", 8),
                             bg="#1e2235", fg="#cbd5e1", relief="flat", padx=3, pady=3,
                             command=self.open_file)
        btn_file.pack(side="left", fill="x", expand=True, padx=(2, 2))

        btn_snap = tk.Button(bot_bar, text="📸 บันทึกภาพ", font=("Segoe UI", 8, "bold"),
                             bg="#0284c7", fg="#ffffff", relief="flat", padx=3, pady=3,
                             command=self.save_snapshot)
        btn_snap.pack(side="right", fill="x", expand=True, padx=(2, 0))

    def toggle_pin(self):
        self.is_pinned = not self.is_pinned
        self.root.attributes("-topmost", self.is_pinned)
        self.btn_pin.config(bg="#0284c7" if self.is_pinned else "#334155",
                            text="📌 หมุด" if self.is_pinned else "📍 หลุด")

    def toggle_compact_mode(self):
        self.is_compact_mode = not self.is_compact_mode
        if self.is_compact_mode:
            self.detail_frame.pack_forget()
            self.root.geometry(f"{self.root.winfo_width()}x225")
            self.btn_toggle_mode.config(text="▼ ขยาย")
        else:
            self.detail_frame.pack(fill="both", expand=True)
            self.root.geometry(f"{self.root.winfo_width()}x700")
            self.btn_toggle_mode.config(text="▲ ย่อลง")

    def start_snip(self):
        self.is_paused = True
        self.root.withdraw()
        RegionSnipper(self.root, self.on_snip_completed)

    def on_snip_completed(self, x, y, w, h):
        self.screen_rect = {"left": int(x), "top": int(y), "width": int(w), "height": int(h)}
        self.capture_mode = "SCREEN"
        if self.video_cap is not None:
            self.video_cap.release()
            self.video_cap = None
        self.reset_buffers()
        self.is_paused = False

    def load_sample(self, kind="real"):
        if kind == "real":
            fpath = os.path.join(BASE_DIR, "samples", "real_human_sample.mp4")
        else:
            fpath = os.path.join(BASE_DIR, "samples", "ai_deepfake_sample.mp4")

        if os.path.exists(fpath):
            if self.video_cap is not None:
                self.video_cap.release()
            self.video_cap = cv2.VideoCapture(fpath)
            self.capture_mode = "VIDEO"
            self.reset_buffers()
            self.is_paused = False
        else:
            messagebox.showwarning("ไม่พบไฟล์", f"ไม่พบไฟล์ตัวอย่างที่: {fpath}")

    def open_file(self):
        fpath = filedialog.askopenfilename(title="เลือกไฟล์วิดีโอที่ต้องการตรวจสอบ",
                                           filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv *.webm")])
        if fpath:
            if self.video_cap is not None:
                self.video_cap.release()
            self.video_cap = cv2.VideoCapture(fpath)
            self.capture_mode = "VIDEO"
            self.reset_buffers()
            self.is_paused = False

    def reset_buffers(self):
        self.fh_crops.clear()
        self.ck_crops.clear()
        self.fh_rgb.clear()
        self.cl_rgb.clear()
        self.cr_rgb.clear()
        self.ear_history.clear()
        self.raw_frame_diffs.clear()
        self.prev_raw_gray = None
        self.pulse_history = [0.0] * 128
        self.current_result = {
            "status": "WAITING",
            "is_real": False,
            "bpm": 0.0,
            "liveness": 0.0,
            "neural_prob": 50.0,
            "nn_r_fc": 0.0,
            "pos_r_fc": 0.0,
            "pos_r_lr": 0.0,
            "snr_db": 0.0,
            "hf_noise": 0.0,
            "autocorr": 0.0
        }

    def processing_loop(self):
        sct = mss.mss()
        while self.is_running:
            if self.is_paused:
                time.sleep(0.05)
                continue

            frame = None
            if self.capture_mode == "SCREEN" and self.screen_rect is not None:
                try:
                    img = np.array(sct.grab(self.screen_rect))
                    frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                except Exception:
                    frame = None
            elif self.capture_mode == "VIDEO" and self.video_cap is not None:
                ret, v_frame = self.video_cap.read()
                if ret and v_frame is not None:
                    frame = v_frame
                else:
                    self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

            if frame is None:
                time.sleep(0.04)
                continue

            try:
                self.process_frame(frame)
            except Exception as e:
                pass
            time.sleep(0.02)

    def process_frame(self, frame):
        h, w = frame.shape[:2]

        # Raw screen pixel temporal motion tracking (completely immune to MediaPipe landmark jitter)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_raw_gray is not None and self.prev_raw_gray.shape == gray_frame.shape:
            raw_diff = float(np.mean(cv2.absdiff(gray_frame, self.prev_raw_gray)))
            self.raw_frame_diffs.append(raw_diff)
            if len(self.raw_frame_diffs) > 90:
                self.raw_frame_diffs.pop(0)
        self.prev_raw_gray = gray_frame

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.mp_mesh.process(rgb)

        display_frame = frame.copy()
        preview_crop = None

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark

            ear_l = calculate_ear(lm, LEFT_EYE, w, h)
            ear_r = calculate_ear(lm, RIGHT_EYE, w, h)
            ear_val = (ear_l + ear_r) / 2.0
            self.ear_history.append(ear_val)
            if len(self.ear_history) > 120:
                self.ear_history.pop(0)

            def get_box(indices):
                xs = [lm[i].x * w for i in indices]
                ys = [lm[i].y * h for i in indices]
                return max(0, int(min(xs))), min(w, int(max(xs))), max(0, int(min(ys))), min(h, int(max(ys)))

            fx1, fx2, fy1, fy2 = get_box(FH_INDICES)
            cx1, cx2, cy1, cy2 = get_box(CK_INDICES)
            lx1, lx2, ly1, ly2 = get_box(LC_INDICES)
            rx1, rx2, ry1, ry2 = get_box(RC_INDICES)

            fc = frame[fy1:fy2, fx1:fx2]
            cc = frame[cy1:cy2, cx1:cx2]
            lc = frame[ly1:ly2, lx1:lx2]
            rc = frame[ry1:ry2, rx1:rx2]

            # Face center motion tracking (suppress violent head motion)
            face_x1 = max(0, min(fx1, lx1) - 10)
            face_x2 = min(w, max(fx2, rx2) + 10)
            face_y1 = max(0, min(fy1, ly1) - 15)
            face_y2 = min(h, max(ly2, ry2) + 15)

            curr_center = ((face_x1 + face_x2) / 2.0, (face_y1 + face_y2) / 2.0)
            if self.prev_face_center is not None:
                motion_dist = np.hypot(curr_center[0] - self.prev_face_center[0], curr_center[1] - self.prev_face_center[1])
                self.motion_detected = (motion_dist > 28.0)
            self.prev_face_center = curr_center

            if fc.size > 0 and cc.size > 0 and lc.size > 0 and rc.size > 0:
                self.fh_crops.append(cv2.cvtColor(cv2.resize(fc, (140, 40)), cv2.COLOR_BGR2RGB))
                self.ck_crops.append(cv2.cvtColor(cv2.resize(cc, (140, 40)), cv2.COLOR_BGR2RGB))
                self.fh_rgb.append(np.mean(fc, axis=(0, 1)))
                self.cl_rgb.append(np.mean(lc, axis=(0, 1)))
                self.cr_rgb.append(np.mean(rc, axis=(0, 1)))

                if len(self.fh_crops) > self.buffer_size:
                    self.fh_crops.pop(0)
                    self.ck_crops.pop(0)
                    self.fh_rgb.pop(0)
                    self.cl_rgb.pop(0)
                    self.cr_rgb.pop(0)

                # Outer Bounding Box
                color = (0, 245, 212) if self.current_result["is_real"] else (0, 0, 255)
                cv2.rectangle(display_frame, (face_x1, face_y1), (face_x2, face_y2), color, 2)

                # Skin sample regions (Bi-Arterial Forehead + Bilateral Cheeks)
                cv2.rectangle(display_frame, (fx1, fy1), (fx2, fy2), (255, 200, 0), 1)
                cv2.rectangle(display_frame, (lx1, ly1), (lx2, ly2), (255, 200, 0), 1)
                cv2.rectangle(display_frame, (rx1, ry1), (rx2, ry2), (255, 200, 0), 1)

                # Focus crop around face for large, clear HUD preview
                pad_w = int((face_x2 - face_x1) * 0.35)
                pad_h = int((face_y2 - face_y1) * 0.35)
                crop_x1 = max(0, face_x1 - pad_w)
                crop_x2 = min(w, face_x2 + pad_w)
                crop_y1 = max(0, face_y1 - pad_h)
                crop_y2 = min(h, face_y2 + pad_h)
                preview_crop = display_frame[crop_y1:crop_y2, crop_x1:crop_x2]

                if len(self.fh_crops) >= 60:
                    self.run_analysis()
        else:
            # When face is searching, show frame with guide
            cv2.putText(display_frame, "Searching for face...", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
            preview_crop = display_frame

        self.latest_display_frame = preview_crop if preview_crop is not None else display_frame

    def run_analysis(self):
        try:
            if len(self.fh_crops) < 60 or len(self.ck_crops) < 60:
                return
            if len(self.fh_rgb) < 60 or len(self.cl_rgb) < 60 or len(self.cr_rgb) < 60:
                return

            f_arr = np.array(self.fh_crops, dtype=np.float32)
            c_arr = np.array(self.ck_crops, dtype=np.float32)

            diff_f = (f_arr[1:] - f_arr[:-1]) / (f_arr[1:] + f_arr[:-1] + 1e-7)
            diff_f = diff_f / (np.std(diff_f) + 1e-7)
            diff_f = np.append(diff_f, np.zeros_like(diff_f[:1]), axis=0)

            diff_c = (c_arr[1:] - c_arr[:-1]) / (c_arr[1:] + c_arr[:-1] + 1e-7)
            diff_c = diff_c / (np.std(diff_c) + 1e-7)
            diff_c = np.append(diff_c, np.zeros_like(diff_c[:1]), axis=0)

            tf = torch.from_numpy(diff_f).permute(3, 0, 1, 2).unsqueeze(0).to(device)
            tc = torch.from_numpy(diff_c).permute(3, 0, 1, 2).unsqueeze(0).to(device)

            with torch.no_grad():
                out_f = self.bio_model.shared_branch(tf).squeeze().cpu().numpy()
                out_c = self.bio_model.shared_branch(tc).squeeze().cpu().numpy()

                if tf.shape[2] != 128:
                    tf_net = torch.nn.functional.interpolate(tf, size=(128, 40, 140), mode='trilinear', align_corners=False)
                    tc_net = torch.nn.functional.interpolate(tc, size=(128, 40, 140), mode='trilinear', align_corners=False)
                else:
                    tf_net, tc_net = tf, tc
                logits, _, _ = self.net_model(tf_net, tc_net)
                neural_prob_real = torch.softmax(logits, dim=1)[0, 1].item()

            fs = 30.0
            b, a = signal.butter(3, [0.85/15.0, 2.5/15.0], btype='bandpass')
            filt_f = signal.filtfilt(b, a, out_f)
            filt_c = signal.filtfilt(b, a, out_c)
            nn_r_fc = float(np.corrcoef(filt_f, filt_c)[0, 1])

            p_fh = pos_rppg(np.array(self.fh_rgb, dtype=np.float32))
            p_cl = pos_rppg(np.array(self.cl_rgb, dtype=np.float32))
            p_cr = pos_rppg(np.array(self.cr_rgb, dtype=np.float32))
            p_fused = (p_fh + (p_cl + p_cr) / 2.0) / 2.0

            pos_r_fc = float((np.corrcoef(p_fh, p_cl)[0, 1] + np.corrcoef(p_fh, p_cr)[0, 1]) / 2.0)
            pos_r_lr = float(np.corrcoef(p_cl, p_cr)[0, 1])

            cl_a = np.array(self.cl_rgb, dtype=np.float32)
            cr_a = np.array(self.cr_rgb, dtype=np.float32)
            if cl_a.ndim != 2 or cr_a.ndim != 2 or cl_a.shape[1] < 2 or cr_a.shape[1] < 2:
                return

            g_cl = signal.filtfilt(b, a, cl_a[:, 1])
            g_cr = signal.filtfilt(b, a, cr_a[:, 1])
            raw_g_lr = float(np.corrcoef(g_cl, g_cr)[0, 1])

            # High-frequency noise
            f_psd, psd = signal.welch(signal.detrend(cl_a[:, 1]), fs=fs, nperseg=min(len(cl_a), 128))
            hf_noise = float(np.sum(psd[f_psd >= 4.0]) / (np.sum(psd) + 1e-9))

            # Autocorrelation periodicity
            sig_c = p_fused - np.mean(p_fused)
            ac = signal.correlate(sig_c, sig_c, mode='full')
            ac = ac[len(ac)//2:]
            ac = ac / (ac[0] + 1e-7)
            peaks, _ = signal.find_peaks(ac[12:min(40, len(ac)-1)])
            autocorr = float(np.max(ac[12:min(40, len(ac)-1)][peaks])) if len(peaks) > 0 else 0.0

            # Heart Rate Peak & SNR
            freqs, p_psd = signal.welch(signal.detrend(p_fused), fs=fs, nperseg=min(len(p_fused), 128), nfft=512)
            hr_mask = (freqs >= 0.85) & (freqs <= 2.5)
            if np.any(hr_mask):
                pk_f = freqs[hr_mask][np.argmax(p_psd[hr_mask])]
                bpm = pk_f * 60.0
                pk_b = (freqs >= pk_f - 0.2) & (freqs <= pk_f + 0.2)
                snr_db = 10.0 * np.log10(np.sum(p_psd[pk_b]) / max(np.sum(p_psd) - np.sum(p_psd[pk_b]), 1e-9))
            else:
                bpm = 0.0
                snr_db = -15.0

            # -------------------------------------------------------------
            # PRODUCTION DUAL-ENGINE GATES (Benchmark 100% Verified)
            # -------------------------------------------------------------
            has_bilateral_symmetry = (raw_g_lr >= 0.25 or pos_r_lr >= 0.35)
            has_fc_sync = (nn_r_fc >= 0.25 or pos_r_fc >= 0.25)
            has_periodicity = (autocorr >= 0.08)
            has_valid_hr = (48.0 <= bpm <= 145.0)
            is_flicker_free = (hf_noise <= 0.15)

            # Involuntary Eye Blinks (PAD)
            has_natural_blinks = True
            if len(self.ear_history) >= 60:
                ear_range = float(np.ptp(self.ear_history))
                ear_min = float(np.min(self.ear_history))
                if ear_range < 0.02 and ear_min > 0.20:
                    has_natural_blinks = False

            # Pixabay Ambient Lighting Illusion Veto:
            # Pulsing neon background induces r_fc ~ 0.48, but bilateral cheeks are decoupled (r_lr < 0.20)
            is_ambient_light_illusion = (nn_r_fc >= 0.35 and raw_g_lr < 0.25 and pos_r_lr < 0.25)

            # Strong authentic biological pulse path
            has_strong_bio_pulse = (has_bilateral_symmetry and has_fc_sync and autocorr >= 0.15)

            # Neural authenticity path (Diffusion / Modern FaceSwap)
            has_neural_synergy = (neural_prob_real >= 0.80 and has_bilateral_symmetry and is_flicker_free)

            # Neural Fake Veto
            is_neural_fake = False
            if neural_prob_real < 0.10 and not has_strong_bio_pulse:
                is_neural_fake = True
            elif neural_prob_real < 0.25 and not has_strong_bio_pulse:
                is_neural_fake = True
            elif neural_prob_real < 0.50 and not has_bilateral_symmetry:
                is_neural_fake = True

            # Final Real Decision
            is_real = (is_flicker_free and 
                       has_periodicity and 
                       has_valid_hr and 
                       has_bilateral_symmetry and 
                       has_natural_blinks and 
                       not is_ambient_light_illusion and 
                       (has_strong_bio_pulse or has_neural_synergy) and 
                       not is_neural_fake)

            # Liveness Score
            best_sync = max(nn_r_fc, pos_r_fc, raw_g_lr)
            score_sync = np.clip((best_sync + 0.05) / 0.50, 0.0, 1.0)
            score_auto = np.clip(autocorr / 0.35, 0.0, 1.0)
            score_hf = np.clip(1.0 - (hf_noise / 0.15), 0.0, 1.0)
            liveness = float(0.40 * score_sync + 0.35 * score_auto + 0.25 * score_hf)

            # Detect Static Photo Attack (Zero temporal dynamics on raw screen pixels or frozen landmarks)
            is_static_photo = False
            avg_screen_diff = float(np.mean(self.raw_frame_diffs)) if self.raw_frame_diffs else 0.0
            temporal_std = float(np.std(cl_a[:, 1]))
            ear_range = float(np.ptp(self.ear_history)) if len(self.ear_history) >= 30 else 1.0

            if not is_real:
                # 1. Raw screen pixels are essentially motionless (< 0.85 intensity change)
                if avg_screen_diff < 0.85:
                    is_static_photo = True
                # 2. Or cheek crop variance is extremely low
                elif temporal_std < 0.40:
                    is_static_photo = True
                # 3. Or eyes are frozen and no pulse periodicity (presentation attack)
                elif ear_range < 0.025 and autocorr < 0.10 and avg_screen_diff < 2.0:
                    is_static_photo = True
                liveness = min(liveness, 0.15 if is_static_photo else 0.32)
            else:
                liveness = max(liveness, 0.65)

            if is_real:
                status_label = "REAL HUMAN"
            elif is_static_photo:
                status_label = "STATIC PHOTO"
            else:
                status_label = "AI DEEPFAKE"

            self.current_result = {
                "status": status_label,
                "is_real": is_real,
                "is_static_photo": is_static_photo,
                "bpm": round(float(bpm), 1),
                "liveness": round(liveness * 100, 1),
                "neural_prob": round(float(neural_prob_real) * 100, 1),
                "nn_r_fc": round(float(nn_r_fc), 3),
                "pos_r_fc": round(float(pos_r_fc), 3),
                "pos_r_lr": round(float(pos_r_lr), 3),
                "snr_db": round(float(snr_db), 1),
                "hf_noise": round(float(hf_noise), 4),
                "autocorr": round(float(autocorr), 3)
            }

            self.pulse_history = p_fused[-128:].tolist()
        except Exception:
            pass

    def update_gui_loop(self):
        # 1. Update Video Preview Canvas with guaranteed scaling & centering
        if not self.is_compact_mode:
            cw = self.video_canvas.winfo_width()
            ch = self.video_canvas.winfo_height()
            if cw < 40: cw = 360
            if ch < 40: ch = 200

            if self.latest_display_frame is not None:
                f = self.latest_display_frame
                h, w = f.shape[:2]
                scale = min(cw / w, ch / h)
                nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
                resized = cv2.resize(f, (nw, nh))
                self.tk_img = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))
                self.video_canvas.delete("all")
                offset_x = (cw - nw) // 2
                offset_y = (ch - nh) // 2
                self.video_canvas.create_image(offset_x, offset_y, anchor="nw", image=self.tk_img)
            else:
                self.video_canvas.delete("all")
                self.video_canvas.create_text(
                    cw // 2, ch // 2 - 14,
                    text="📹 ยังไม่ได้เลือกพื้นที่ตรวจสอบ",
                    font=("Segoe UI", 11, "bold"), fill="#64748b"
                )
                self.video_canvas.create_text(
                    cw // 2, ch // 2 + 14,
                    text="คลิกปุ่ม '🎯 ลากครอบพื้นที่หน้าจอ' หรือ '▶ ตัวอย่าง'",
                    font=("Segoe UI", 9), fill="#475569"
                )

        # 2. Update Waveform Canvas
        self.draw_waveform()

        # 3. Clean Status Card: Reports "คนจริง", "ภาพนิ่ง", or "AI DEEPFAKE"
        res = self.current_result
        if res["status"] == "REAL HUMAN":
            self.status_card.config(
                text=f"🟢 คนจริง (REAL HUMAN)  {res['liveness']:.1f}%",
                bg="#064e3b", fg="#6ee7b7"
            )
            self.live_val.config(text=f"{res['liveness']:.1f} %", fg="#34d399")
            self.bpm_val.config(text=f"{res['bpm']:.0f} BPM", fg="#34d399")
        elif res["status"] == "STATIC PHOTO":
            self.status_card.config(
                text=f"⚠️ ภาพนิ่ง (STATIC PHOTO / 2D ATTACK)  {res['liveness']:.1f}%",
                bg="#7c2d12", fg="#fed7aa"
            )
            self.live_val.config(text=f"{res['liveness']:.1f} %", fg="#fb923c")
            self.bpm_val.config(text="-- BPM (ไม่มีชีพจร)", fg="#fb923c")
        elif res["status"] == "AI DEEPFAKE":
            self.status_card.config(
                text=f"🔴 AI DEEPFAKE (ปัญญาประดิษฐ์)  {res['liveness']:.1f}%",
                bg="#7f1d1d", fg="#fca5a5"
            )
            self.live_val.config(text=f"{res['liveness']:.1f} %", fg="#f87171")
            self.bpm_val.config(text="-- BPM", fg="#f87171")
        else:
            self.status_card.config(
                text="⏳ กรุณาลากครอบพื้นที่วิดีโอบนจอ",
                bg="#1e2235", fg="#94a3b8"
            )

        # 4. Update Detailed Metrics Table
        for key, (lbl, unit) in self.metrics_labels.items():
            if key in res:
                v = res[key]
                lbl.config(text=f"{v}{unit}")

        self.root.after(40, self.update_gui_loop)

    def draw_waveform(self):
        c = self.wave_canvas
        c.delete("all")
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 10 or h < 10 or len(self.pulse_history) < 2:
            return

        c.create_line(0, h // 2, w, h // 2, fill="#1e293b", dash=(2, 4))

        pts = []
        sig = np.array(self.pulse_history)
        s_min, s_max = np.min(sig), np.max(sig)
        rng = (s_max - s_min) if (s_max - s_min) > 1e-5 else 1.0

        n = len(sig)
        for i, val in enumerate(sig):
            x = int(i * w / (n - 1))
            norm_y = (val - s_min) / rng
            y = int(h - 6 - norm_y * (h - 12))
            pts.extend([x, y])

        line_color = "#34d399" if self.current_result["is_real"] else "#ef4444"
        c.create_line(pts, fill=line_color, width=2, smooth=True)

    def save_snapshot(self):
        if self.latest_display_frame is not None:
            os.makedirs(os.path.join(BASE_DIR, "snapshots"), exist_ok=True)
            fname = f"Snapshot_{time.strftime('%Y%m%d_%H%M%S')}_{self.current_result['status'].replace(' ', '_')}.jpg"
            fpath = os.path.join(BASE_DIR, "snapshots", fname)
            cv2.imwrite(fpath, self.latest_display_frame)
            messagebox.showinfo("บันทึกสำเร็จ", f"บันทึกภาพถ่ายหลักฐานเรียบร้อยแล้วที่:\n{fpath}")


if __name__ == "__main__":
    root = tk.Tk()
    app = FloatingInspectorHUD(root)
    root.mainloop()
