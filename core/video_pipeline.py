"""
core/video_pipeline.py

Fast video deepfake detection pipeline.

Detection strategy:
  1. Sample MAX_FRAMES evenly across the video
  2. Use OpenCV Haar Cascade (built-in, ~10-30ms/frame) to find faces — NO RetinaFace
  3. Crop each face region and run the ViT deepfake detector
  4. Aggregate probabilities (mean + max) across frames

Speed: ~5-15 seconds for a typical 30-second video on CPU.
"""

import cv2
import numpy as np
from PIL import Image

# Haar cascade path — always available in OpenCV
HAAR_XML = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

MAX_FRAMES     = 6    # Maximum frames to analyse
MIN_FACE_PX    = 60   # Ignore faces smaller than this (px)
RESIZE_WIDTH   = 640  # Downscale wide frames before cascade


class VideoPipeline:
    """
    Fast video deepfake detection pipeline using Haar cascade + ViT detector.
    Falls back to full-frame analysis if no face is found in a frame.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.face_cascade = cv2.CascadeClassifier(HAAR_XML)
        # Lazy-imported to avoid loading at server startup
        self._detector = None

    def _get_detector(self):
        if self._detector is None:
            from core.deepfake_detector import DeepfakeDetector
            self._detector = DeepfakeDetector(device=self.device)
        return self._detector

    # ------------------------------------------------------------------
    def _sample_frame_indices(self, total_frames: int) -> list[int]:
        """Return MAX_FRAMES evenly-spaced frame indices."""
        n = min(MAX_FRAMES, total_frames)
        if n <= 1:
            return [0]
        step = total_frames / n
        return [int(i * step) for i in range(n)]

    def _detect_face_crop(self, bgr_frame: np.ndarray):
        """
        Detect largest face using Haar cascade and return a PIL crop.
        Returns None if no face detected (caller will use full frame).
        """
        # Downscale for faster cascade
        h, w = bgr_frame.shape[:2]
        if w > RESIZE_WIDTH:
            scale = RESIZE_WIDTH / w
            small = cv2.resize(bgr_frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            small, scale = bgr_frame, 1.0

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(MIN_FACE_PX, MIN_FACE_PX),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if len(faces) == 0:
            return None

        # Pick the largest face
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        # Scale back to original resolution
        x, y, fw, fh = (
            int(x / scale), int(y / scale),
            int(fw / scale), int(fh / scale),
        )

        # Add 20% padding around face
        pad_x = int(fw * 0.20)
        pad_y = int(fh * 0.20)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(bgr_frame.shape[1], x + fw + pad_x)
        y2 = min(bgr_frame.shape[0], y + fh + pad_y)

        crop = bgr_frame[y1:y2, x1:x2]
        rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    # ------------------------------------------------------------------
    def process_video(self, video_path: str, fps_extraction: int = 1) -> dict | None:
        """
        Analyse a video file and return aggregated deepfake probabilities.

        Returns:
            {
                'fake_prob':    float (0-100),
                'real_prob':    float (0-100),
                'prediction':   'FAKE' | 'REAL',
                'confidence':   float (0-100),
                'frames_analysed': int,
                'faces_found':     int,
                'method':          str,
                'frame_scores':    list[float],   # per-frame fake_prob 0-1
            }
            or None if the video cannot be opened / read.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            # Some containers don't report frame count — read first to get it
            total_frames = 300  # assume 10s @ 30fps as fallback

        sample_indices = set(self._sample_frame_indices(total_frames))

        detector = self._get_detector()
        frame_fake_probs: list[float] = []
        faces_found = 0
        frame_idx   = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx in sample_indices:
                face_pil = self._detect_face_crop(frame)

                if face_pil is not None:
                    faces_found += 1
                    result = detector.predict_pil(face_pil)
                else:
                    # No face detected — run on the full (downscaled) frame
                    h, w = frame.shape[:2]
                    if w > RESIZE_WIDTH:
                        frame_small = cv2.resize(
                            frame,
                            (RESIZE_WIDTH, int(h * RESIZE_WIDTH / w)),
                            interpolation=cv2.INTER_AREA,
                        )
                    else:
                        frame_small = frame
                    result = detector.predict_bgr(frame_small)

                frame_fake_probs.append(result["fake_prob"] / 100.0)

                # Early exit
                if len(frame_fake_probs) >= MAX_FRAMES:
                    break

            frame_idx += 1

        cap.release()

        if not frame_fake_probs:
            return None

        mean_fake = float(np.mean(frame_fake_probs))
        max_fake  = float(np.max(frame_fake_probs))

        # Weighted aggregate: 70% mean + 30% max (catches partial fakes)
        agg_fake = 0.70 * mean_fake + 0.30 * max_fake

        prediction = "FAKE" if agg_fake >= 0.5 else "REAL"
        confidence = agg_fake * 100 if agg_fake >= 0.5 else (1 - agg_fake) * 100

        return {
            "fake_prob":       round(agg_fake * 100, 2),
            "real_prob":       round((1 - agg_fake) * 100, 2),
            "prediction":      prediction,
            "confidence":      round(confidence, 2),
            "frames_analysed": len(frame_fake_probs),
            "faces_found":     faces_found,
            "method":          "video_deepfake_vit",
            "frame_scores":    [round(p * 100, 2) for p in frame_fake_probs],
        }
