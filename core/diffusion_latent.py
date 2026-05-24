import torch
import torch.nn as nn

try:
    from diffusers import AutoencoderTiny
except ImportError:
    AutoencoderTiny = None

class DiffusionErrorLoop(nn.Module):
    """
    Module 4 Helper: Latent/Diffusion Fingerprint Branch (Pillar C)
    Extracts the absolute error formulation: E = |I_input - I_recon|
    using a frozen pre-trained VAE from Stable Diffusion.
    """
    def __init__(self, model_id="madebyollin/taesd", device="cuda"):
        super().__init__()
        self.device = device
        if AutoencoderTiny is None:
            raise ImportError("The 'diffusers' library is required. Install via 'pip install diffusers'")
            
        # Load pre-trained Tiny VAE (TAESD) and freeze it
        # This uses < 200MB VRAM and is 50x faster than the original SD VAE
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.vae = AutoencoderTiny.from_pretrained(model_id, torch_dtype=dtype).to(self.device)
        self.vae.eval()
        
        for param in self.vae.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        """
        x: Normalized image tensor [B, 3, H, W] in the range [0, 1]
        Returns the error map E = |I_input - I_recon|
        """
        # VAE typically expects input in [-1, 1]
        x_scaled = x * 2.0 - 1.0
        
        # Move to same dtype as VAE
        x_scaled = x_scaled.to(dtype=self.vae.dtype, device=self.device)
        
        # 1. Compress to latent space representation z
        # TAESD returns .latents directly instead of a distribution
        z = self.vae.encode(x_scaled).latents
        
        # 2. Inject a minimal, deterministic noise coefficient (t = 0.05)
        # Note: True diffusion forward step requires adding noise according to a schedule.
        # Here we approximate by adding a small scaled Gaussian noise.
        noise = torch.randn_like(z)
        t = 0.05
        z_noisy = z + t * noise
        
        # 3. Single-step backward reconstruction execution
        I_recon_scaled = self.vae.decode(z_noisy).sample
        
        # Convert back to [0, 1] range for both
        I_recon = (I_recon_scaled / 2.0) + 0.5
        I_recon = I_recon.clamp(0, 1)
        x_orig = (x_scaled / 2.0) + 0.5
        
        # 4. Absolute Error Formulation
        error_map = torch.abs(x_orig - I_recon)
        
        # Return in float32 for downstream processing
        return error_map.to(torch.float32)

