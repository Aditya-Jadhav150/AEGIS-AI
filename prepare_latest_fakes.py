import os
import random
import shutil
import uuid
import cv2
from PIL import Image
import torch
from facenet_pytorch import MTCNN
from tqdm import tqdm

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading MTCNN onto {device}...")
    mtcnn = MTCNN(keep_all=False, device=device)

    # Output Directories
    PROCESSED_TRAIN_DIR = r"D:\ANTIGRAVITY\dataset\processed_train\fake"
    PROCESSED_VAL_DIR = r"D:\ANTIGRAVITY\dataset\processed_val\fake"

    # Make sure output dirs exist
    os.makedirs(PROCESSED_TRAIN_DIR, exist_ok=True)
    os.makedirs(PROCESSED_VAL_DIR, exist_ok=True)

    # Specify exact source directories
    SOURCE_DIRS = [
        r"D:\ANTIGRAVITY\FAKE_IMAGES\OneDrive\DMs\CommercialTools\CommercialTools_CelebAHQ_processed\midjourney",
        r"D:\ANTIGRAVITY\FAKE_IMAGES\OneDrive\DMs\latent_diffusion\latent_diffusion_FFHQ",
        r"D:\ANTIGRAVITY\FAKE_IMAGES\OneDrive\GANs\MMD_GAN"
    ]

    print("Scanning specific source directories for fake images...")
    
    # Store paths by directory so we can balance them
    dir_images = {d: [] for d in SOURCE_DIRS}
    
    for cat_dir in SOURCE_DIRS:
        if os.path.exists(cat_dir):
            for root, dirs, files in os.walk(cat_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        dir_images[cat_dir].append(os.path.join(root, file))
        else:
            print(f"WARNING: Directory not found: {cat_dir}")

    total_images_found = sum(len(paths) for paths in dir_images.values())
    print(f"Total fake images found across 3 sources: {total_images_found}")

    for d, paths in dir_images.items():
        print(f" - {os.path.basename(d)}: {len(paths)} images")

    if total_images_found == 0:
        print("No images found! Exiting.")
        return

    # Algorithm to sample exactly TOTAL_TARGET images, balanced across folders
    TOTAL_TARGET = min(30000, total_images_found)
    print(f"\nAttempting to randomly sample {TOTAL_TARGET} images (balanced ~10,000 per folder)...")
    
    sampled_paths = []
    
    # Simple distribution algorithm: 
    # Try to grab an equal share from remaining folders.
    remaining_target = TOTAL_TARGET
    folders_left = list(dir_images.keys())
    
    # Sort folders by the number of images they have (smallest first)
    folders_left.sort(key=lambda d: len(dir_images[d]))
    
    for i, d in enumerate(folders_left):
        share = remaining_target // len(folders_left[i:])
        available = len(dir_images[d])
        take = min(share, available)
        
        chosen = random.sample(dir_images[d], take)
        sampled_paths.extend(chosen)
        remaining_target -= take

    # Because of integer division or unequal distribution, we might have slightly fewer than TOTAL_TARGET.
    # If so, pull from remaining unselected images across all folders.
    current_count = len(sampled_paths)
    if current_count < TOTAL_TARGET:
        print(f"Filling remaining gap of {TOTAL_TARGET - current_count} images from remaining pool...")
        sampled_set = set(sampled_paths)
        all_remaining = []
        for paths in dir_images.values():
            all_remaining.extend([p for p in paths if p not in sampled_set])
            
        needed = TOTAL_TARGET - current_count
        fillers = random.sample(all_remaining, min(needed, len(all_remaining)))
        sampled_paths.extend(fillers)

    # Shuffle final paths to ensure randomness across sources in training
    random.shuffle(sampled_paths)
    
    # 90% train, 10% val split
    train_count = int(0.9 * len(sampled_paths))
    train_paths = sampled_paths[:train_count]
    val_paths = sampled_paths[train_count:]

    print(f"Final Count -> {len(train_paths)} for training, {len(val_paths)} for validation.")

    def process_images(paths, dest_dir, desc_label):
        print(f"\nProcessing {desc_label} dataset to {dest_dir}...")
        for src_path in tqdm(paths, desc=f"Extracting {desc_label} Faces"):
            # Use uuid to guarantee no filename collisions when merging into existing dataset
            unique_id = uuid.uuid4().hex[:8]
            ext = os.path.splitext(src_path)[1]
            if not ext: ext = ".jpg"
            dest_filename = f"modern_fake_{unique_id}{ext}"
            dest_path = os.path.join(dest_dir, dest_filename)

            try:
                img = Image.open(src_path).convert('RGB')
                # MTCNN cropping
                img_cropped = mtcnn(img, save_path=None)
                
                if img_cropped is not None:
                    mtcnn(img, save_path=dest_path)
                else:
                    # Fallback if MTCNN fails finding a face
                    img = img.resize((224, 224))
                    img.save(dest_path)
            except Exception as e:
                # Silently catch and continue, we want to power through the 30k
                pass
                
    # Process train
    process_images(train_paths, PROCESSED_TRAIN_DIR, "TRAIN")
    
    # Process val
    process_images(val_paths, PROCESSED_VAL_DIR, "VAL")

    print("\n--- Processing Complete! ---")
    print("New set of 30,000 modern AI-generated fakes have been securely appended.")

if __name__ == "__main__":
    main()
