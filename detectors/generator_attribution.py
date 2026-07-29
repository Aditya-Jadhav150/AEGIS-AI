import torch
from PIL import Image

class GeneratorAttributionHead:
    """
    Auxiliary forensic module. Given an image classified as AI-generated,
    estimates the likely source generator.
    
    IMPORTANT: This prediction is PROBABILISTIC.
    """
    
    GENERATORS = [
        'Stable Diffusion XL',
        'Stable Diffusion 3',
        'Flux',
        'Midjourney',
        'Adobe Firefly',
        'Ideogram',
        'DALL-E 3',
        'StyleGAN2/3',
        'Unknown Generator'
    ]
    
    def __init__(self, device='cpu'):
        self.device = device
        self._model = None
        self._processor = None
    
    def _lazy_load(self):
        if self._model is None:
            # Placeholder for actual model loading (e.g., fine-tuned ConvNeXt-Tiny)
            # using torchvision or timm
            import torchvision.models as models
            import torchvision.transforms as transforms
            
            self._model = models.resnet18(weights=None)
            self._model.fc = torch.nn.Linear(self._model.fc.in_features, len(self.GENERATORS))
            self._model.to(self.device)
            self._model.eval()
            
            self._processor = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def predict(self, image_path: str) -> dict:
        """
        Returns top 3 probable generators with confidence scores.
        """
        self._lazy_load()
        try:
            image = Image.open(image_path).convert("RGB")
            tensor = self._processor(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits = self._model(tensor)
                probs = torch.nn.functional.softmax(logits, dim=-1)[0]
                
            top_probs, top_indices = torch.topk(probs, 3)
            results = []
            for i in range(3):
                results.append({
                    "generator": self.GENERATORS[top_indices[i]],
                    "confidence": float(top_probs[i])
                })
            return {"top_3": results}
        except Exception as e:
            print(f"Generator Attribution Failed: {e}")
            return {"top_3": [{"generator": "Unknown Generator", "confidence": 1.0}]}
