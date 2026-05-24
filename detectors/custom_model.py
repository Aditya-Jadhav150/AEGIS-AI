import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

class CustomModelDetector:
    def __init__(self, model_id="prithivMLmods/Deep-Fake-Detector-v2-Model", device=None):
        """
        Initializes the custom secondary model.
        :param model_id: HuggingFace Hub model ID or path to local checkpoint.
        :param device: torch.device instance.
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"Loading Secondary Custom Model [{model_id}] onto {self.device}...")
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_id)
            self.model = AutoModelForImageClassification.from_pretrained(model_id)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Note: torch.compile() is disabled here because Triton is not supported on Windows natively.
            # The model will still run extremely fast using FP16 and standard CUDA eager execution.
            
        except Exception as e:
            print(f"Failed to load secondary model: {e}")
            self.model = None

    def analyze(self, image_pil):
        """
        Analyzes the image using the custom secondary model.
        :param image_pil: PIL Image.
        :return: dict with 'score' (0 to 1) where 1 is Fake, and 'confidence'.
        """
        if self.model is None:
            return {"score": 0.5, "confidence": 0.0, "error": "Model not loaded"}

        try:
            inputs = self.processor(images=image_pil, return_tensors="pt").to(self.device)
            
            # Apply FP16 mixed precision for inference speedup
            with torch.no_grad(), torch.autocast(device_type=self.device.type):
                outputs = self.model(**inputs)
                logits = outputs.logits
                model_probs = torch.nn.functional.softmax(logits, dim=1)
                
            labels = self.model.config.id2label
            fake_prob = 0.0
            real_prob = 0.0
            
            for idx, label_name in labels.items():
                prob = model_probs[0][idx].item()
                l = label_name.lower()
                if 'fake' in l or 'deepfake' in l or 'spoof' in l:
                    fake_prob += prob
                elif 'real' in l or 'pristine' in l:
                    real_prob += prob
                    
            if fake_prob == 0 and real_prob == 0:
                pred_idx = torch.argmax(model_probs, dim=1).item()
                predicted_label = labels[pred_idx].lower()
                if 'fake' in predicted_label:
                    fake_prob = 1.0
                else:
                    real_prob = 1.0
                    
            return {
                "score": float(fake_prob),
                "confidence": 0.85, # Strong confidence for the custom ViT
            }
            
        except Exception as e:
            print(f"Custom Model Analysis Error: {e}")
            return {"score": 0.5, "confidence": 0.0, "error": str(e)}

if __name__ == "__main__":
    detector = CustomModelDetector(device=torch.device('cpu'))
    from PIL import Image
    import numpy as np
    dummy_img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    print(detector.analyze(dummy_img))
