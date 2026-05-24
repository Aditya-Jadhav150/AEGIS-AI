import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    import timm
except ImportError:
    timm = None

class SpatialDomainBranch(nn.Module):
    """
    Module 2: Spatial Domain Branch (Pillar A)
    Captures localized facial mutations, blending artifacts at splice interfaces, and macroscopic anatomical inaccuracies.
    """
    def __init__(self, d_out=512):
        super().__init__()
        if timm is None:
            raise ImportError("The 'timm' library is required for SpatialDomainBranch. Install it via 'pip install timm'")
        # Swin-T Backbone
        self.backbone = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0)
        self.proj = nn.Linear(self.backbone.num_features, d_out)
        
    def forward(self, x):
        # x is the spatial_tensor [B, 3, 512, 512]
        # Resize from 512x512 to 224x224 for Swin-T
        x_resized = F.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)
        features = self.backbone(x_resized)
        return self.proj(features)

class FrequencyDomainBranch(nn.Module):
    """
    Module 3: Frequency Domain Branch (Pillar B)
    Captures generative upsampling footprints and high-frequency noise injection profiles.
    """
    def __init__(self, d_out=512):
        super().__init__()
        import torchvision.models as models
        # ResNet18 configured with 1-channel input
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Remove original fully connected layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.proj = nn.Linear(num_ftrs, d_out)

    def forward(self, spectrum):
        # spectrum is the pre-computed log-magnitude 2D FFT [B, 1, 512, 512]
        features = self.resnet(spectrum)
        return self.proj(features)

class LatentFingerprintBranch(nn.Module):
    """
    Module 4: Latent/Diffusion Fingerprint Branch (Pillar C)
    Processes the absolute error formulation: E = |I_input - I_recon|
    """
    def __init__(self, d_out=512):
        super().__init__()
        # Three 2D convolutional blocks with max-pooling
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((4, 4))
        )
        self.proj = nn.Linear(128 * 4 * 4, d_out)

    def forward(self, error_map):
        # error_map is the pre-computed latent_tensor [B, 3, 512, 512]
        features = self.cnn(error_map)
        features = features.view(features.size(0), -1)
        return self.proj(features)

class StatisticalRealismBranch(nn.Module):
    """
    Module 5: Realism Consistency & Statistical Branch (Pillar D)
    Uses Variance & Laplacians to find synthetic coherence.
    """
    def __init__(self, d_out=256):
        super().__init__()
        # Two-layer MLP with Layer Normalization
        self.mlp = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, d_out),
            nn.LayerNorm(d_out)
        )

    def forward(self, stat_tensor):
        # stat_tensor is the pre-computed true LBP/Entropy tensor [B, 8192]
        return self.mlp(stat_tensor)

class CrossAttentionFusion(nn.Module):
    """
    Module 6: Dynamic Attention-Based Feature Fusion
    Multi-Head Cross-Attention layer to weight the branches dynamically.
    """
    def __init__(self, d_spatial=512, d_freq=512, d_latent=512, d_stat=256, d_model=512):
        super(CrossAttentionFusion, self).__init__()
        
        # Project heterogeneous feature vectors into a uniform dimensional subspace
        self.proj_spatial = nn.Linear(d_spatial, d_model)
        self.proj_freq    = nn.Linear(d_freq, d_model)
        self.proj_latent  = nn.Linear(d_latent, d_model)
        self.proj_stat    = nn.Linear(d_stat, d_model)
        
        # Instantiate Multihead Attention Layer
        self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=8, batch_first=True)
        
        # Classification MLP Layer Structure (Module 7)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Sigmoid is executed by loss function or final layer
        )

    def forward(self, f_spatial, f_freq, f_latent, f_stat, return_features=False):
        # Establish query vector from Spatial Structural Core
        q = self.proj_spatial(f_spatial).unsqueeze(1) # Tensor format: [Batch, 1, d_model]
        
        # Standardize alternate domain features to function as Keys and Values
        k_freq = self.proj_freq(f_freq).unsqueeze(1)
        k_latent = self.proj_latent(f_latent).unsqueeze(1)
        k_stat = self.proj_stat(f_stat).unsqueeze(1)
        
        # Stack domain keys across sequence dimension bounds
        kv = torch.cat([k_freq, k_latent, k_stat], dim=1) # Tensor format: [Batch, 3, d_model]
        
        # Execute Cross-Attention Analysis pass
        attn_output, _ = self.multihead_attn(query=q, key=kv, value=kv)
        attn_output = attn_output.squeeze(1) # Collapse shape back to: [Batch, d_model]
        
        # Complete downstream classification mapping
        logits = self.mlp(attn_output)
        
        if return_features:
            return logits, q.squeeze(1), kv
        return logits

class MultiModalDeepfakeSystemV2(nn.Module):
    """
    Integrated R&D Production Specification V2.0
    Multi-Modal Deepfake Face Detection System
    """
    def __init__(self):
        super().__init__()
        self.spatial_branch = SpatialDomainBranch()
        self.freq_branch = FrequencyDomainBranch()
        self.latent_branch = LatentFingerprintBranch()
        self.stat_branch = StatisticalRealismBranch()
        self.fusion_head = CrossAttentionFusion()

    def forward(self, spatial_tensor, freq_tensor, latent_tensor, stat_tensor, return_features=False):
        """
        Receives purely pre-computed offline tensors to eliminate online bottlenecks.
        """
        f_spatial = self.spatial_branch(spatial_tensor)
        f_freq = self.freq_branch(freq_tensor)
        f_latent = self.latent_branch(latent_tensor)
        f_stat = self.stat_branch(stat_tensor)
        
        if return_features:
            logits, q, kv = self.fusion_head(f_spatial, f_freq, f_latent, f_stat, return_features=True)
            # Pool features for contrastive loss
            return logits, q
        
        logits = self.fusion_head(f_spatial, f_freq, f_latent, f_stat)
        return logits

class CompoundLoss(nn.Module):
    """
    Multi-Objective Compound Loss Formulation:
    L_total = L_BCE + lambda * L_Contrastive
    """
    def __init__(self, lambda_weight=0.35, temperature=0.07):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.lambda_weight = lambda_weight
        self.temperature = temperature

    def forward(self, logits, labels, features):
        loss_bce = self.bce_loss(logits.view(-1), labels.float())
        
        features = F.normalize(features, dim=1)
        batch_size = features.shape[0]
        
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        mask_self = torch.eye(batch_size, device=labels.device)
        mask_positive = mask - mask_self
        
        max_sim, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        exp_sim = torch.exp(sim_matrix - max_sim)
        
        exp_sim_sum = exp_sim.sum(dim=1, keepdim=True) - torch.exp(sim_matrix - max_sim) * mask_self
        log_prob = sim_matrix - max_sim - torch.log(exp_sim_sum + 1e-8)
        
        mean_log_prob_pos = (mask_positive * log_prob).sum(1) / (mask_positive.sum(1) + 1e-8)
        loss_contrastive = -mean_log_prob_pos.mean()
        
        return loss_bce + self.lambda_weight * loss_contrastive, loss_bce, loss_contrastive
