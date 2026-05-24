import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image

# Import the new V2 Architecture Modules
from core.v2_architecture import MultiModalDeepfakeSystemV2, CompoundLoss
from core.diffusion_latent import DiffusionErrorLoop

class MultiModalDataset(datasets.DatasetFolder):
    def __init__(self, root):
        # Only look for .pt files
        super().__init__(root, loader=torch.load, extensions=('.pt',))

    def __getitem__(self, index):
        path, _ = self.samples[index]
        # data is a dict: {spatial_tensor, freq_tensor, latent_tensor, stat_tensor, label}
        data = self.loader(path)
        return data

def validate(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            spatial = batch["spatial_tensor"].to(device)
            freq = batch["freq_tensor"].to(device)
            latent = batch["latent_tensor"].to(device)
            stat = batch["stat_tensor"].to(device)
            labels = batch["label"].to(device)
            
            # Forward Main Architecture
            outputs = model(spatial, freq, latent, stat)
            
            # Threshold logit at 0.0 (equivalent to prob > 0.5)
            predicted = (outputs.squeeze() > 0.0).long()
            total += labels.size(0)
            correct += (predicted == labels.squeeze()).sum().item()
    return 100 * correct / total

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)
    
    # Enable NVIDIA CuDNN Auto-Tuner: Drastically speeds up convolution math on fixed-size images
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    train_data = MultiModalDataset("dataset/processed_train")
    val_data = MultiModalDataset("dataset/processed_val")

    print("Class mapping:", train_data.class_to_idx)

    # VAE Error loop and Multi-Modal model are heavy, lowering batch size from 8 to 4
    batch_size = 4
    accumulation_steps = 8

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True 
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True
    )

    print("Initializing Multi-Modal Deepfake System V2...")
    model = MultiModalDeepfakeSystemV2().to(device)
    
    # Calculate class weights to handle dataset imbalance
    # targets may not be immediately available depending on DatasetFolder processing
    # but self.targets is populated in DatasetFolder
    labels_list = train_data.targets
    class_counts = np.bincount(labels_list)
    if len(class_counts) > 1:
        # pos_weight = negative_samples / positive_samples (class 0 / class 1)
        pos_weight = torch.tensor([class_counts[0] / class_counts[1]], device=device, dtype=torch.float)
    else:
        pos_weight = None

    # We use our new Compound Loss (BCE + Contrastive)
    criterion = CompoundLoss(lambda_weight=0.35)
    if pos_weight is not None:
        criterion.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

    epochs = 10
    best_val_acc = 0.0
    patience = 5
    patience_counter = 0

    scaler = torch.amp.GradScaler('cuda')

    try:
        for epoch in range(epochs):
            print(f"Starting Epoch {epoch+1}...")
            model.train()
            total_loss = 0
            
            optimizer.zero_grad()

            for i, batch in enumerate(train_loader):
                if i == 0 and epoch == 0:
                    print("SUCCESS: Grabbed the first batch of pre-computed tensors. Processing...")

                spatial = batch["spatial_tensor"].to(device)
                freq = batch["freq_tensor"].to(device)
                latent = batch["latent_tensor"].to(device)
                stat = batch["stat_tensor"].to(device)
                labels = batch["label"].to(device)

                with torch.amp.autocast('cuda'):
                    # Forward Main Architecture, returning features for Contrastive Loss
                    logits, features = model(spatial, freq, latent, stat, return_features=True)
                    
                    # Compute Compound Loss
                    loss, loss_bce, loss_contrastive = criterion(logits, labels, features)
                    loss = loss / accumulation_steps

                scaler.scale(loss).backward()

                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                total_loss += loss.item() * accumulation_steps
                
                # Print periodic batch updates
                if i % 10 == 0:
                    print(f"Batch {i}/{len(train_loader)} - BCE: {loss_bce.item():.4f}, Contrastive: {loss_contrastive.item():.4f}")

            avg_loss = total_loss / len(train_loader)
            val_acc = validate(model, val_loader, device)

            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val Accuracy: {val_acc:.2f}%")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), "model_best.pth")
                print("--> Best model checkpoint completely secured (model_best.pth).")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break
                
            scheduler.step()

    except KeyboardInterrupt:
        print("\n[!] Training halted manually by user. The highest accuracy checkpoint is completely saved as 'model_best.pth'!")
    
    torch.save(model.state_dict(), "model.pth")
    print("\nTraining procedure officially terminated. 'model.pth' securely written to disk!")

if __name__ == "__main__":
    main()