import torch
from PIL import Image

class GeneralAIImageDetector:
    """
    Detects AI-generated images regardless of content (faces, landscapes, art, objects).
    The specific backend model is selected via benchmarking.
    
    The detector is loaded lazily on first prediction to minimize cold start impact.
    """
    
    def __init__(self, model_name: str = "umm-maybe/AI-image-detector", device: str = 'cpu'):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._processor = None
    
    def _lazy_load(self):
        """Defer model loading until first prediction to optimize cold start."""
        if self._model is None:
            # Depending on the model_name, load the appropriate model pipeline.
            # Using ViTForImageClassification as a default placeholder for HuggingFace models.
            try:
                from transformers import ViTForImageClassification, ViTImageProcessor
                # Example HuggingFace pipeline. Adjust according to the benchmarked model.
                self._processor = ViTImageProcessor.from_pretrained(self.model_name)
                self._model = ViTForImageClassification.from_pretrained(self.model_name).to(self.device)
                self._model.eval()
            except ImportError:
                raise ImportError("Please install transformers and torch to use the GeneralAIImageDetector.")

    def predict(self, image_path: str) -> float:
        """
        Returns the probability that the image is AI-generated (0.0 to 1.0).
        """
        self._lazy_load()
        
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self._processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                
                # Assume the model maps label 1 to "AI" and label 0 to "Real"
                # (Needs to be adjusted based on the specific model's label mapping)
                # Typically, umm-maybe/AI-image-detector has 0: artificial, 1: human
                # Let's map it accordingly. If we need prob of AI generated:
                # If label 0 is Artificial (AI), then prob is probs[0][0]
                # We will return the probability of class 0 for this specific model
                ai_prob = probs[0][0].item()
                return float(ai_prob)
        except Exception as e:
            # Fallback if prediction fails
            print(f"Warning: General AI Image Detector failed: {e}")
            return 0.5
