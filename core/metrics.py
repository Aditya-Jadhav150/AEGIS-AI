import torch
import cv2
import numpy as np
from skimage.measure import shannon_entropy
from scipy.stats import kurtosis

class ForensicMetricExtractor:
    """
    Unified extraction interface for the 9 production forensic metrics.
    Accepts aligned face tensors and raw image arrays.
    Returns a dict of 9 named scalar float values.
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        # Import dynamically to prevent cyclic dependencies if they import metrics
        from core.diffusion_latent import DiffusionErrorLoop
        from core.statistical_extraction import StatisticalFeatureExtractor
        
        self.error_loop = DiffusionErrorLoop(device=self.device)
        self.stat_extractor = StatisticalFeatureExtractor()

    def compute_spatial_score(self, aligned_tensor: torch.Tensor) -> float:
        """Mean of absolute values of normalized face tensor."""
        return float(torch.mean(torch.abs(aligned_tensor)).item())

    def compute_freq_score(self, aligned_tensor: torch.Tensor) -> float:
        """High-frequency energy from outer 25% of FFT magnitude spectrum."""
        gray_tensor = 0.2989 * aligned_tensor[0:1, :, :] + 0.5870 * aligned_tensor[1:2, :, :] + 0.1140 * aligned_tensor[2:3, :, :]
        gray_tensor = gray_tensor.unsqueeze(0)  # Shape: [1, 1, 512, 512]
        freq_complex = torch.fft.fft2(gray_tensor)
        freq_shifted = torch.fft.fftshift(torch.abs(freq_complex), dim=(-2, -1))
        freq_tensor = torch.log(1 + freq_shifted)
        
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

    def compute_latent_score(self, aligned_tensor: torch.Tensor) -> float:
        """Mean absolute TAESD VAE reconstruction error."""
        latent_err = self.error_loop(aligned_tensor.unsqueeze(0))
        return float(torch.mean(torch.abs(latent_err)).item())

    def compute_stat_score(self, aligned_tensor: torch.Tensor) -> float:
        """Mean of local variance + edge feature vector."""
        stat_tensor = self.stat_extractor(aligned_tensor.unsqueeze(0)).cpu()
        return float(torch.mean(stat_tensor).item())

    def compute_entropy(self, image_array: np.ndarray) -> float:
        """Shannon entropy of grayscale face crop."""
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        return float(shannon_entropy(gray))

    def compute_edge_density(self, image_array: np.ndarray) -> float:
        """Canny edge pixel ratio (thresholds 100, 200)."""
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return float(np.sum(edges > 0) / edges.size)

    def compute_laplacian_variance(self, image_array: np.ndarray) -> float:
        """Laplacian operator variance — measures blurriness."""
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return float(lap.var())

    def compute_color_kurtosis(self, image_array: np.ndarray) -> float:
        """Average kurtosis across R, G, B channels."""
        ks = [kurtosis(image_array[..., c].ravel()) for c in range(3)]
        return float(np.mean(ks))

    def compute_jpeg_consistency(self, image_array: np.ndarray) -> float:
        """DCT coefficient variance excluding low-frequency 8x8 block."""
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY).astype(np.float32)
        dct_full = cv2.dct(gray)
        mask = np.ones_like(dct_full, dtype=bool)
        mask[:8, :8] = False
        return float(np.var(dct_full[mask]))

    def extract_all(self, aligned_tensor: torch.Tensor, image_array: np.ndarray) -> dict:
        """Returns dict of all 9 metrics as named floats."""
        return {
            "spatial_score": self.compute_spatial_score(aligned_tensor),
            "freq_score": self.compute_freq_score(aligned_tensor),
            "latent_score": self.compute_latent_score(aligned_tensor),
            "stat_score": self.compute_stat_score(aligned_tensor),
            "entropy": self.compute_entropy(image_array),
            "edge_density": self.compute_edge_density(image_array),
            "laplacian_variance": self.compute_laplacian_variance(image_array),
            "color_kurtosis": self.compute_color_kurtosis(image_array),
            "jpeg_consistency": self.compute_jpeg_consistency(image_array),
        }
