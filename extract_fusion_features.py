# -*- coding: utf-8 -*-
"""extract_fusion_features.py

Iterates over the .pt tensors produced by `data_pipeline.py` and extracts a
compact set of forensic scores for each image. The original four scores are
kept (spatial, frequency, latent, statistical) and five lightweight statistical
features are added:
    * Entropy
    * Edge density
    * Laplacian variance
    * Color kurtosis
    * JPEG‑consistency (block‑DCT variance)

The resulting CSV (`dataset/fusion_features.csv`) now has **9 feature columns**
plus the label column, ready for training the XGBoost Fusion Engine.
"""

import os
import torch
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm

# ----- Additional statistical utilities -----
from skimage.measure import shannon_entropy
from scipy.stats import kurtosis


def compute_entropy(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
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
    # Compute kurtosis per channel, then average
    ks = [kurtosis(img_rgb[..., c].ravel()) for c in range(3)]
    return float(np.mean(ks))


def compute_jpeg_consistency(img_rgb):
    # Approximate JPEG consistency by computing DCT of the whole grayscale image
    # and measuring variance of the high‑frequency coefficients (excluding the low‑freq 8×8 corner).
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # Apply 2‑D DCT to the entire image
    dct_full = cv2.dct(gray)
    # Create mask to ignore low‑frequency top‑left 8×8 region
    mask = np.ones_like(dct_full, dtype=bool)
    mask[:8, :8] = False
    # Variance of the remaining coefficients serves as a JPEG‑consistency proxy
    return float(np.var(dct_full[mask]))


def high_freq_energy(freq_tensor: torch.Tensor) -> float:
    # Same helper used in the inference script – sum of outer‑25% energy
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


def main():
    processed_root = os.path.join('dataset', 'processed_train')
    # Gather all .pt files (including sharded sub‑folders)
    pt_files = []
    for dirpath, _, filenames in os.walk(processed_root):
        for f in filenames:
            if f.lower().endswith('.pt'):
                pt_files.append(os.path.join(dirpath, f))

    print(f"Found {len(pt_files)} .pt files to process.")
    rows = []
    for pt_path in tqdm(pt_files, desc="Extracting features"):
        data = torch.load(pt_path, map_location='cpu')
        # Original tensors
        spatial = data.get('spatial_tensor')
        freq = data.get('freq_tensor')
        latent = data.get('latent_tensor')
        stat = data.get('stat_tensor')
        label = data.get('label')

        # ---- Core scores (same as before) ----
        spatial_score = torch.mean(torch.abs(spatial)).item() if spatial is not None else 0.0
        # Frequency – high‑frequency energy
        if freq is not None:
            freq_score = high_freq_energy(freq)
        else:
            freq_score = 0.0
        latent_score = torch.mean(torch.abs(latent)).item() if latent is not None else 0.0
        stat_score = torch.mean(stat.float()).item() if stat is not None else 0.0

        # ---- Convert the aligned tensor back to a NumPy RGB image for the new descriptors ----
        # Denormalize: pixel = (normalized * std + mean) * 255
        if spatial is not None:
            mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            spatial_np = spatial.cpu().numpy()
            unnorm = (spatial_np * std + mean) * 255.0
            rgb = np.clip(unnorm, 0, 255).transpose(1, 2, 0).astype(np.uint8)
        else:
            rgb = np.zeros((512, 512, 3), dtype=np.uint8)

        # ---- New lightweight descriptors ----
        entropy_score = compute_entropy(rgb)
        edge_density_score = compute_edge_density(rgb)
        laplacian_var_score = compute_laplacian_variance(rgb)
        color_kurtosis_score = compute_color_kurtosis(rgb)
        jpeg_consistency_score = compute_jpeg_consistency(rgb)

        # ---- Assemble row ----
        label_val = int(label.item()) if isinstance(label, torch.Tensor) else int(label)
        row = {
            "spatial_score": spatial_score,
            "freq_score": freq_score,
            "latent_score": latent_score,
            "stat_score": stat_score,
            "entropy": entropy_score,
            "edge_density": edge_density_score,
            "laplacian_variance": laplacian_var_score,
            "color_kurtosis": color_kurtosis_score,
            "jpeg_consistency": jpeg_consistency_score,
            "label": label_val,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    output_csv = os.path.join('dataset', 'fusion_features.csv')
    df.to_csv(output_csv, index=False)
    print(f"Feature CSV saved to {output_csv} (shape: {df.shape})")

if __name__ == "__main__":
    main()
