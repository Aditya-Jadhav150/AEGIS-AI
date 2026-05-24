import torch
import torch.nn as nn
import torch.nn.functional as F

class StatisticalFeatureExtractor(nn.Module):
    """
    Computes local texture randomness and structural edges completely on the GPU.
    Bypasses CPU skimage completely for ultra-fast throughput.
    """
    def __init__(self, kernel_size=9):
        super().__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        # Laplacian kernel for fast edge extraction
        self.register_buffer('laplacian_kernel', torch.tensor([[[[0.0,  1.0, 0.0], 
                                                                  [1.0, -4.0, 1.0], 
                                                                  [0.0,  1.0, 0.0]]]]))
        
    def forward(self, x):
        """
        x: Normalized image tensor [B, 3, 512, 512] on GPU
        Returns features: [B, 512]
        """
        # Convert to Grayscale natively on GPU
        gray = 0.2989 * x[:, 0:1, :, :] + 0.5870 * x[:, 1:2, :, :] + 0.1140 * x[:, 2:3, :, :]
        
        # Calculate E[X] (Local Mean) via an optimized uniform box pool
        local_mean = F.avg_pool2d(gray, kernel_size=self.kernel_size, stride=1, padding=self.pad)
        
        # Calculate E[X^2] (Local Mean of Squares)
        local_mean_sq = F.avg_pool2d(gray ** 2, kernel_size=self.kernel_size, stride=1, padding=self.pad)
        
        # Variance = E[X^2] - (E[X])^2
        local_variance = local_mean_sq - (local_mean ** 2)
        local_variance = torch.clamp(local_variance, min=0.0)
        
        # Extract structural edges via fast GPU Laplacian
        edge_map = torch.abs(F.conv2d(gray, self.laplacian_kernel, padding=1))
        
        # Concatenate and flatten
        features = torch.cat([
            F.adaptive_avg_pool2d(local_variance, (16, 16)).flatten(1),
            F.adaptive_avg_pool2d(edge_map, (16, 16)).flatten(1)
        ], dim=1)
        
        return features # Shape: [B, 512]
