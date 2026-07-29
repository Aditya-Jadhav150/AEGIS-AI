"""
core/deepfake_detector.py

Primary deepfake detection engine using a pre-trained HuggingFace model.
Model: dima806/deepfake_vs_real_image_detection (ViT-based, trained specifically
for face deepfake detection on large datasets).

This replaces the XGBoost pipeline as the PRIMARY verdict engine.
The XGBoost forensic metrics remain as supplementary dashboard signals.
"""

import torch
from PIL import Image
import numpy as np
import cv2


class DeepfakeDetector:
    """
    Lazy-loaded deepfake detection engine backed by a ViT model
    specifically fine-tuned for real vs. deepfake face classification.

    Model: dima806/deepfake_vs_real_image_detection
    Labels: 0 = Deepfake, 1 = Real (model-specific)
    """

    MODEL_NAME = "dima806/deepfake_vs_real_image_detection"

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return
        from transformers import pipeline as hf_pipeline
        self._pipe = hf_pipeline(
            "image-classification",
            model=self.MODEL_NAME,
            device=0 if self.device == "cuda" and torch.cuda.is_available() else -1,
        )

    def predict_pil(self, pil_image: Image.Image) -> dict:
        """
        Run inference on a PIL Image.

        Returns:
            {
                'fake_prob':  float (0-1),
                'real_prob':  float (0-1),
                'prediction': 'FAKE' | 'REAL',
                'confidence': float (0-100),
                'method':     str,
            }
        """
        self._load()
        results = self._pipe(pil_image)
        # Build a label → score dict regardless of ordering
        scores = {r["label"].lower(): r["score"] for r in results}

        # The model uses labels "Deepfake" and "Real"
        fake_prob = scores.get("deepfake", scores.get("fake", 1 - scores.get("real", 0.5)))
        real_prob = 1.0 - fake_prob

        prediction = "FAKE" if fake_prob >= 0.5 else "REAL"
        confidence = fake_prob * 100 if fake_prob >= 0.5 else real_prob * 100

        return {
            "fake_prob": round(fake_prob * 100, 2),
            "real_prob": round(real_prob * 100, 2),
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "method": "deepfake_vit",
        }

    def predict_path(self, image_path: str) -> dict:
        """Run inference from a file path."""
        img = Image.open(image_path).convert("RGB")
        return self.predict_pil(img)

    def predict_bgr(self, bgr_frame: np.ndarray) -> dict:
        """Run inference on an OpenCV BGR numpy array."""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        return self.predict_pil(pil)
