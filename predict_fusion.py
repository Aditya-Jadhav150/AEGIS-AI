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
# Helper functions – same as used in the FastAPI service
# --------------------------------------------------------------
def high_freq_energy(freq_tensor: torch.Tensor) -> float:
    freq = freq_tensor.squeeze().cpu()
    h, w = freq.shape
    mask = torch.zeros_like(freq, dtype=torch.bool)
    margin_h = h // 4
    margin_w = w // 4
    mask[:margin_h, :] = True
    mask[-margin_h:, :] = True
    mask[:, :margin_w] = True
    mask[:, -margin_w:] = True
    return float(torch.sum(torch.abs(freq)[mask]))

def compute_entropy(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    from skimage.measure import shannon_entropy
    return float(shannon_entropy(gray))

def compute_edge_density(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return float(np.sum(edges > 0) / edges.size)

def compute_laplacian_variance(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())

def compute_color_kurtosis(img_rgb):
    from scipy.stats import kurtosis
    k = [kurtosis(img_rgb[..., c].ravel()) for c in range(3)]
    return float(np.mean(k))

def compute_jpeg_consistency(img_rgb):
    # Approximate JPEG consistency by computing DCT of the whole grayscale image
    # and measuring variance of the high‑frequency coefficients (excluding low‑freq 8×8 corner).
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # Apply 2‑D DCT to the entire image
    dct_full = cv2.dct(gray)
    # Mask out low‑frequency top‑left 8×8 region
    mask = np.ones_like(dct_full, dtype=bool)
    mask[:8, :8] = False
    # Return variance of remaining coefficients as proxy
    return float(np.var(dct_full[mask]))

# --------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run forensic inference on a single image")
    parser.add_argument("--image", "-i", required=True, help="Path to image file")
    parser.add_argument("--model", default=os.path.join('dataset', 'fusion_engine_best.json'), help="Path to saved XGBoost model JSON")
    parser.add_argument("--scaler", default=os.path.join('dataset', 'scaler.json'), help="Path to saved StandardScaler JSON")
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
    aligner = GeometricAligner(device='cpu')
    aligned = aligner.align_and_crop(bgr, return_tensor=True)
    if aligned is None:
        raise RuntimeError("Face detection failed – no face found in the image.")

    # -------------------- Compute original four scores --------------------
    spatial_score = float(torch.mean(torch.abs(aligned)).item())
    # Frequency score (exactly matching data_pipeline.py on the normalized aligned tensor)
    gray_tensor = 0.2989 * aligned[0:1, :, :] + 0.5870 * aligned[1:2, :, :] + 0.1140 * aligned[2:3, :, :]
    gray_tensor = gray_tensor.unsqueeze(0)  # Shape: [1, 1, 512, 512]
    freq_complex = torch.fft.fft2(gray_tensor)
    freq_shifted = torch.fft.fftshift(torch.abs(freq_complex), dim=(-2, -1))
    freq_tensor = torch.log(1 + freq_shifted)
    freq_score = high_freq_energy(freq_tensor)
    # Latent error (TAESD auto‑encoder)
    error_loop = DiffusionErrorLoop(device='cpu')
    latent_err = error_loop(aligned.unsqueeze(0))
    latent_score = float(torch.mean(torch.abs(latent_err)).item())
    # Statistical embedding (LBP / entropy tensor)
    stat_extractor = StatisticalFeatureExtractor()
    stat_tensor = stat_extractor(aligned.unsqueeze(0)).cpu()
    stat_score = float(torch.mean(stat_tensor).item())

    # -------------------- New lightweight descriptors --------------------
    # Denormalize: pixel = (normalized * std + mean) * 255
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    aligned_np = aligned.cpu().numpy()
    unnorm = (aligned_np * std + mean) * 255.0
    rgb = np.clip(unnorm, 0, 255).transpose(1, 2, 0).astype(np.uint8)

    entropy_score = compute_entropy(rgb)
    edge_density_score = compute_edge_density(rgb)
    laplacian_var_score = compute_laplacian_variance(rgb)
    color_kurtosis_score = compute_color_kurtosis(rgb)
    jpeg_consistency_score = compute_jpeg_consistency(rgb)

    # -------------------- Assemble feature vector --------------------
    feature_dict = {
        "spatial_score": spatial_score,
        "freq_score": freq_score,
        "latent_score": latent_score,
        "stat_score": stat_score,
        "entropy": entropy_score,
        "edge_density": edge_density_score,
        "laplacian_variance": laplacian_var_score,
        "color_kurtosis": color_kurtosis_score,
        "jpeg_consistency": jpeg_consistency_score,
    }
    df_feat = pd.DataFrame([feature_dict])
    # Scale using the same scaler that was fit on training data
    df_scaled = scaler.transform(df_feat)

    # -------------------- Predict --------------------
    prob_fake = model.predict_proba(df_scaled)[0, 1]
    verdict = "AI Generated" if prob_fake >= 0.5 else "Real"
    confidence = prob_fake * 100 if prob_fake >= 0.5 else (1 - prob_fake) * 100

    # -------------------- Pretty report --------------------
    print("\n" + "=" * 40)
    print("          AI IMAGE FORENSIC REPORT")
    print("=" * 40 + "\n")
    print(f"Spatial Artifact Score   : {spatial_score:.2f}")
    print(f"Frequency Anomaly Score  : {freq_score:.2f}")
    print(f"Noise Residual Score     : {latent_score:.2f}")
    print(f"Embedding Consistency    : {stat_score:.2f}")
    print(f"Entropy Score            : {entropy_score:.2f}")
    print(f"Edge Density Score       : {edge_density_score:.2f}")
    print(f"Laplacian Variance Score : {laplacian_var_score:.2f}")
    print(f"Color Kurtosis Score     : {color_kurtosis_score:.2f}")
    print(f"JPEG Consistency Score   : {jpeg_consistency_score:.2f}")
    print("-" * 40)
    print("FINAL RESULT:")
    print(f"Likely {verdict}")
    print(f"Confidence: {confidence:.0f}%")
    print("=" * 40 + "\n")

if __name__ == "__main__":
    main()
