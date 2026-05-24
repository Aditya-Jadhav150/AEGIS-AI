import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from core.v2_architecture import MultiModalDeepfakeSystemV2

class MultiModalDataset(datasets.DatasetFolder):
    def __init__(self, root):
        # Only look for .pt files
        super().__init__(root, loader=torch.load, extensions=('.pt',))

    def __getitem__(self, index):
        path, _ = self.samples[index]
        data = self.loader(path)
        return data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_data = MultiModalDataset("dataset/processed_test")
test_loader = DataLoader(test_data, batch_size=4, num_workers=4, pin_memory=True)

print("Loading Multi-Modal Deepfake System V2...")
model = MultiModalDeepfakeSystemV2().to(device)
try:
    model.load_state_dict(torch.load("model_best.pth", map_location=device, weights_only=True))
    print("Successfully loaded model_best.pth")
except FileNotFoundError:
    print("model_best.pth not found, attempting to load model.pth")
    model.load_state_dict(torch.load("model.pth", map_location=device, weights_only=True))

model.eval()

y_true = []
y_pred = []

print("Evaluating...")
with torch.no_grad():
    for batch in test_loader:
        spatial = batch["spatial_tensor"].to(device)
        freq = batch["freq_tensor"].to(device)
        latent = batch["latent_tensor"].to(device)
        stat = batch["stat_tensor"].to(device)
        labels = batch["label"].to(device)
        
        # Forward Main Architecture
        outputs = model(spatial, freq, latent, stat)
        
        # Binary Classification from Logits (threshold at 0)
        preds = (outputs.squeeze() > 0.0).long()

        y_true.extend(labels.squeeze().cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

print(classification_report(y_true, y_pred))