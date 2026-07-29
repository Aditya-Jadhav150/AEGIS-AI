# -*- coding: utf-8 -*-
"""predict_fusion.py

CLI inference script for the Hybrid AI Image Forensics system.
It loads the persisted XGBoost model and StandardScaler, extracts the nine
scalar forensic scores from a single image, and prints a polished report.
"""

import os
import argparse
import joblib
import pandas as pd
import torch
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler

# Core components – already in the repo
from core.alignment import GeometricAligner
from core.diffusion_latent import DiffusionErrorLoop
from core.statistical_extraction import StatisticalFeatureExtractor

# --------------------------------------------------------------
# Helper functions removed: utilizing core.metrics
# --------------------------------------------------------------
from core.metrics import ForensicMetricExtractor
from core.metadata_forensics import MetadataForensicsEngine

def main():
    parser = argparse.ArgumentParser(description="Run forensic inference on a single image")
    parser.add_argument("--image", "-i", required=True, help="Path to image file")
    parser.add_argument("--model", default=os.path.join('dataset', 'fusion_engine_best.json'), help="Path to saved XGBoost model JSON")
    parser.add_argument("--scaler", default=os.path.join('dataset', 'scaler.json'), help="Path to saved StandardScaler JSON")
    parser.add_argument("--device", default="cpu", help="Device to run models on (cpu or cuda)")
    args = parser.parse_args()

    # Load model & scaler
    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")
    if not os.path.isfile(args.scaler):
        raise FileNotFoundError(f"Scaler file not found: {args.scaler}")
        
    from xgboost import XGBClassifier
    import json
    
    model = XGBClassifier()
    model.load_model(args.model)
    
    scaler = StandardScaler()
    with open(args.scaler, 'r') as f:
        s_data = json.load(f)
    scaler.mean_ = np.array(s_data["mean"])
    scaler.var_ = np.array(s_data["var"])
    scaler.scale_ = np.array(s_data["scale"])
    scaler.n_features_in_ = s_data["n_features_in"]

    # -------------------- Load image & align --------------------
    bgr = cv2.imread(args.image)
    if bgr is None:
        raise ValueError(f"Could not read image: {args.image}")
    aligner = GeometricAligner(device=args.device)
    aligned = aligner.align_and_crop(bgr, return_tensor=True)
    if aligned is None:
        raise RuntimeError("Face detection failed – no face found in the image.")

    # -------------------- Load Metric Extractor --------------------
    metric_extractor = ForensicMetricExtractor(device=args.device)
    metadata_engine = MetadataForensicsEngine()
    
    # -------------------- Extract Metadata Forensics ---------------
    metadata_results = metadata_engine.analyze(args.image)
    meta_score = metadata_results.get('metadata_forensic_score', 0.5)

    # -------------------- Denormalize image --------------------
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    aligned_np = aligned.cpu().numpy()
    unnorm = (aligned_np * std + mean) * 255.0
    rgb = np.clip(unnorm, 0, 255).transpose(1, 2, 0).astype(np.uint8)

    # -------------------- Compute scores --------------------
    feature_dict = metric_extractor.extract_all(aligned, rgb)
    # Add metadata forensic score to feature dict
    feature_dict['metadata_forensic_score'] = meta_score
    
    df_feat = pd.DataFrame([feature_dict])
    # The scaler expects specific columns in the same order as training.
    # The scaler natively expects 10 features now.
    df_scaled = scaler.transform(df_feat)

    # -------------------- Predict --------------------
    prob_fake = model.predict_proba(df_scaled)[0, 1]
    verdict = "AI Generated" if prob_fake >= 0.5 else "Real"
    confidence = prob_fake * 100 if prob_fake >= 0.5 else (1 - prob_fake) * 100

    # -------------------- Pretty report --------------------
    print("\n" + "=" * 40)
    print("          AI IMAGE FORENSIC REPORT")
    print("=" * 40 + "\n")
    print(f"Spatial Artifact Score   : {feature_dict['spatial_score']:.2f}")
    print(f"Frequency Anomaly Score  : {feature_dict['freq_score']:.2f}")
    print(f"Noise Residual Score     : {feature_dict['latent_score']:.2f}")
    print(f"Embedding Consistency    : {feature_dict['stat_score']:.2f}")
    print(f"Entropy Score            : {feature_dict['entropy']:.2f}")
    print(f"Edge Density Score       : {feature_dict['edge_density']:.2f}")
    print(f"Laplacian Variance Score : {feature_dict['laplacian_variance']:.2f}")
    print(f"Color Kurtosis Score     : {feature_dict['color_kurtosis']:.2f}")
    print(f"JPEG Consistency Score   : {feature_dict['jpeg_consistency']:.2f}")
    print("-" * 40)
    print("FINAL RESULT:")
    print(f"Likely {verdict}")
    print(f"Confidence: {confidence:.0f}%")
    print("=" * 40 + "\n")

if __name__ == "__main__":
    main()
