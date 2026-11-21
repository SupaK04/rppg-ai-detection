"""
ROI Extractor for Siamese-rPPG Network.
Extracts Forehead and Cheek Regions of Interest (ROIs) from video frames.
Matches specifications in Tsou et al. (2020):
- Forehead ROI: Top 1/3 of the detected face region
- Cheek ROI: Center/middle 1/3 of the face region
- Resized to standard size (H=40, W=140)
"""

from typing import Tuple, List, Optional
import os
import cv2
import numpy as np
import torch


class ROIExtractor:
    def __init__(
        self,
        target_size: Tuple[int, int] = (140, 40),  # (width, height)
        haar_cascade_path: Optional[str] = None
    ):
        """
        Args:
            target_size: (width, height) for output ROIs. Default is (140, 40).
            haar_cascade_path: Optional path to OpenCV haar cascade xml.
        """
        self.target_width, self.target_height = target_size

        if haar_cascade_path and os.path.exists(haar_cascade_path):
            self.face_cascade = cv2.CascadeClassifier(haar_cascade_path)
        else:
            default_cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if os.path.exists(default_cascade):
                self.face_cascade = cv2.CascadeClassifier(default_cascade)
            else:
                self.face_cascade = None

    def detect_face(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detects primary frontal face in frame.
        Returns:
            (x, y, w, h) bounding box, or None if not found.
        """
        if self.face_cascade is None:
            # Fallback: Central region heuristic
            h, w = frame.shape[:2]
            return int(w * 0.25), int(h * 0.2), int(w * 0.5), int(h * 0.6)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )
        if len(faces) == 0:
            h, w = frame.shape[:2]
            return int(w * 0.25), int(h * 0.2), int(w * 0.5), int(h * 0.6)

        # Pick largest face
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        return tuple(faces[0])

    def get_rois_from_face(
        self,
        frame: np.ndarray,
        face_bbox: Tuple[int, int, int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crops forehead and cheek regions from given face bounding box.
        - Forehead: upper 1/3 of face
        - Cheek: middle 1/3 of face
        Returns:
            (forehead_roi, cheek_roi) both resized to (target_height, target_width, 3)
        """
        x, y, w, h = face_bbox
        frame_h, frame_w = frame.shape[:2]

        # Ensure bounds
        x = max(0, min(x, frame_w - 1))
        y = max(0, min(y, frame_h - 1))
        w = max(1, min(w, frame_w - x))
        h = max(1, min(h, frame_h - y))

        one_third_h = int(h / 3.0)

        # Forehead: top 1/3 (y to y + one_third_h), width inset slightly to avoid hair
        x_inset = int(w * 0.1)
        forehead_x = x + x_inset
        forehead_y = y
        forehead_w = max(1, w - 2 * x_inset)
        forehead_h = max(1, one_third_h)

        forehead_crop = frame[forehead_y:forehead_y + forehead_h, forehead_x:forehead_x + forehead_w]

        # Cheek: middle 1/3 (y + one_third_h to y + 2*one_third_h)
        cheek_x = x + x_inset
        cheek_y = y + one_third_h
        cheek_w = max(1, w - 2 * x_inset)
        cheek_h = max(1, one_third_h)

        cheek_crop = frame[cheek_y:cheek_y + cheek_h, cheek_x:cheek_x + cheek_w]

        return self._resize_crop(forehead_crop), self._resize_crop(cheek_crop)

    def _resize_crop(self, crop: np.ndarray) -> np.ndarray:
        if crop.size == 0:
            return np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        return cv2.resize(crop, (self.target_width, self.target_height))

    def extract_from_video(
        self,
        video_path: str,
        num_frames: Optional[int] = None,
        stride: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts forehead and cheek ROI sequence from video file.
        Returns:
            v_forehead: (1, 3, T, 40, 140) torch FloatTensor normalized to [0, 1]
            v_cheek:    (1, 3, T, 40, 140) torch FloatTensor normalized to [0, 1]
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {video_path}")

        forehead_list: List[np.ndarray] = []
        cheek_list: List[np.ndarray] = []

        face_bbox = None
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % stride == 0:
                # Detect face on first sampled frame
                if face_bbox is None:
                    face_bbox = self.detect_face(frame)

                f_roi, c_roi = self.get_rois_from_face(frame, face_bbox)
                # Convert BGR to RGB
                f_roi_rgb = cv2.cvtColor(f_roi, cv2.COLOR_BGR2RGB)
                c_roi_rgb = cv2.cvtColor(c_roi, cv2.COLOR_BGR2RGB)

                forehead_list.append(f_roi_rgb)
                cheek_list.append(c_roi_rgb)

                if num_frames and len(forehead_list) >= num_frames:
                    break

            frame_idx += 1

        cap.release()

        if len(forehead_list) == 0:
            raise ValueError(f"No frames extracted from {video_path}")

        # Convert to numpy arrays: shape (T, H, W, C)
        f_arr = np.array(forehead_list, dtype=np.float32) / 255.0
        c_arr = np.array(cheek_list, dtype=np.float32) / 255.0

        # Permute to PyTorch format: (1, C, T, H, W)
        v_forehead = torch.from_numpy(f_arr).permute(3, 0, 1, 2).unsqueeze(0)
        v_cheek = torch.from_numpy(c_arr).permute(3, 0, 1, 2).unsqueeze(0)

        return v_forehead, v_cheek
