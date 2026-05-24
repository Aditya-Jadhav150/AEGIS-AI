import os
import glob
import random
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


from core.alignment import GeometricAligner
from core.diffusion_latent import DiffusionErrorLoop
from core.statistical_extraction import StatisticalFeatureExtractor

class RawImageDataset(Dataset):
    """ Purely handles multi-threaded Disk I/O. Zero model weights loaded here. """
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image_bgr = cv2.imread(img_path)
            if image_bgr is None: 
                return None
            return {"path": img_path, "image": image_bgr}
        except Exception:
            return None

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    paths = [b["path"] for b in batch]
    images = [b["image"] for b in batch]
    return paths, images

class DataPipeline:
    def __init__(self, raw_data_dir="dataset/raw", processed_data_dir="dataset/processed", batch_size=32):
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Initialize MTCNN on the GPU in the main thread (CRITICAL FIX)
        self.aligner = GeometricAligner(device=self.device)
        
        # 2. Initialize VAE explicitly on CUDA and force Half-Precision
        self.error_loop = DiffusionErrorLoop(device=self.device)
        self.error_loop.vae = self.error_loop.vae.to(self.device).half()
        
        # 3. Statistical Extractor on GPU
        self.stat_extractor = StatisticalFeatureExtractor().to(self.device)
        
        os.makedirs(self.processed_data_dir, exist_ok=True)
        for label in ["real", "fake"]:
            os.makedirs(os.path.join(self.processed_data_dir, label), exist_ok=True)

    def ingest_dataset(self, source_dir, label, max_images=5000):
        if label not in ["real", "fake"]:
            raise ValueError("Label must be 'real' or 'fake'")
            
        print(f"Starting ingestion from {source_dir} as {label}...")
        
        extensions = ('*.jpg', '*.jpeg', '*.png', '*.webp')
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(source_dir, ext)))
            image_files.extend(glob.glob(os.path.join(source_dir, ext.upper())))
            
        # Optional subset sampling to save disk space
        if max_images is not None and len(image_files) > max_images:
            print(f"Found {len(image_files)} images. Randomly sampling {max_images} to save disk space...")
            random.seed(42) # For reproducibility between runs if needed
            image_files = random.sample(image_files, max_images)
        else:
            print(f"Found {len(image_files)} images.")
        
        dataset = RawImageDataset(image_files)
        
        # DataLoader optimized for Windows local infrastructure
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=0, 
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        success_count = 0
        global_idx = 0
        label_val = 0.0 if label == "real" else 1.0
        
        for batch in tqdm(dataloader, desc="Batched CUDA Offline Processing"):
            if batch is None:
                continue
                
            paths, images_bgr = batch
            
            valid_paths = []
            spatial_list = []
            
            # Execute MTCNN alignment on GPU sequentially but extremely fast
            for path, img_bgr in zip(paths, images_bgr):
                aligned_rgb = self.aligner.align_and_crop(img_bgr, return_tensor=False)
                if aligned_rgb is not None:
                    tensor_img = torch.from_numpy(aligned_rgb).permute(2, 0, 1).float() / 255.0
                    spatial = self.aligner.normalize(tensor_img)
                    spatial_list.append(spatial)
                    valid_paths.append(path)
                    
            if not valid_paths:
                continue
                
            spatial_tensor = torch.stack(spatial_list).to(self.device, non_blocking=True)
            
            # 1. GPU Feature Extraction
            stat_tensor = self.stat_extractor(spatial_tensor)
            
            # 2. Fast GPU Fourier Transforms
            gray = 0.2989 * spatial_tensor[:, 0:1, :, :] + 0.5870 * spatial_tensor[:, 1:2, :, :] + 0.1140 * spatial_tensor[:, 2:3, :, :]
            fft2 = torch.fft.fft2(gray)
            fft2_shifted = torch.fft.fftshift(fft2, dim=(-2, -1))
            magnitude = torch.abs(fft2_shifted)
            freq_tensor = torch.log(1 + magnitude)
            
            # 3. Stable Diffusion VAE processing (Forced Half-Precision)
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    latent_tensor = self.error_loop(spatial_tensor)
                    
            # Serialize
            spatial_cpu = spatial_tensor.cpu()
            freq_cpu = freq_tensor.cpu()
            latent_cpu = latent_tensor.cpu().float()
            stat_cpu = stat_tensor.cpu()
            
            for i in range(len(valid_paths)):
                batch_data = {
                    "spatial_tensor": spatial_cpu[i].clone(),
                    "freq_tensor": freq_cpu[i].clone(),
                    "latent_tensor": latent_cpu[i].clone(),
                    "stat_tensor": stat_cpu[i].clone(),
                    "label": torch.tensor([label_val], dtype=torch.float)
                }
                name, _ = os.path.splitext(os.path.basename(valid_paths[i]))
                
                # NTFS Directory Sharding: Prevent folder indexing throttle
                chunk_id = global_idx // 1000
                chunk_dir = os.path.join(self.processed_data_dir, label, f"shard_{chunk_id}")
                os.makedirs(chunk_dir, exist_ok=True)
                
                save_path = os.path.join(chunk_dir, f"{name}.pt")
                torch.save(batch_data, save_path)
                
                global_idx += 1
                
            success_count += len(valid_paths)
            
            del spatial_tensor, freq_tensor, latent_tensor, stat_tensor
            torch.cuda.empty_cache()
                    
        print(f"\nSuccessfully processed {success_count} out of {len(image_files)} images.")

if __name__ == "__main__":
    # Force CUDA algorithm optimization
    torch.backends.cudnn.benchmark = True
    print(f"Is CUDA available? {torch.cuda.is_available()}")
    print(f"Current GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
    
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Modal Offline Preprocessing Pipeline")
    parser.add_argument("--source", type=str, required=True, help="Directory containing raw images")
    parser.add_argument("--label", type=str, choices=["real", "fake"], required=True, help="Label for the images")
    parser.add_argument("--output", type=str, default="dataset/processed", help="Output directory")
    parser.add_argument("--max_images", type=int, default=5000, help="Maximum number of images to process (to save disk space)")
    
    args = parser.parse_args()
    
    pipeline = DataPipeline(processed_data_dir=args.output)
    pipeline.ingest_dataset(args.source, args.label, max_images=args.max_images)
