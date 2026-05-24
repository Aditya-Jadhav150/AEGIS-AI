import requests
import json
import base64
from io import BytesIO
import time
import os

# Attempt to load .env automatically
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))
except ImportError:
    # Manual fallback if python-dotenv is not installed
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, val = line.strip().split('=', 1)
                    os.environ[key.strip()] = val.strip()

class ExternalAPIDetector:
    def __init__(self, api_endpoint=None, api_key=None, provider=None):
        """
        Initializes the External API Detector.
        :param api_endpoint: The endpoint URL.
        :param api_key: The authentication key.
        :param provider: 'nvidia_nim' or 'huggingface' or 'mock'
        """
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.provider = provider
        
        # Check environment variables for NVIDIA NIM key
        env_nim_key = os.environ.get('NVIDIA_NIM_API_KEY')
        if env_nim_key and not self.api_key:
            self.api_key = env_nim_key
            self.provider = "nvidia_nim"
            if not self.api_endpoint:
                self.api_endpoint = "https://ai.api.nvidia.com/v1/cv/hive/deepfake-image-detection"
                
        # If no endpoint/key provided, default to mock
        if not self.api_endpoint or not self.api_key:
            self.provider = "mock"

    def analyze(self, image_pil):
        """
        Sends the image to the external API for deepfake detection.
        :param image_pil: PIL Image.
        :return: dict with 'score' (0 to 1), 'confidence', and 'label'.
        """
        if self.provider == "mock":
            return self._mock_analyze(image_pil)
        
        # Prepare image bytes
        buffered = BytesIO()
        image_pil.save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()
        
        if self.provider == "huggingface":
            return self._analyze_huggingface(img_bytes)
        elif self.provider == "nvidia_nim":
            return self._analyze_nvidia_nim(img_bytes)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _analyze_huggingface(self, img_bytes):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            response = requests.post(self.api_endpoint, headers=headers, data=img_bytes)
            response.raise_for_status()
            result = response.json()
            
            # HuggingFace typically returns a list of dicts like: [{'label': 'fake', 'score': 0.9}, ...]
            fake_score = 0.0
            real_score = 0.0
            for item in result:
                label = item['label'].lower()
                if 'fake' in label or 'spoof' in label:
                    fake_score += item['score']
                elif 'real' in label or 'pristine' in label:
                    real_score += item['score']
                    
            return {
                "score": float(fake_score), # Probability of being fake
                "confidence": 0.9,          # External APIs are usually treated with high confidence
                "provider": self.provider
            }
        except Exception as e:
            print(f"HuggingFace API Error: {e}")
            return self._mock_analyze(None, error=str(e))

    def _analyze_nvidia_nim(self, img_bytes):
        # Implementation for NVIDIA NIM API (build.nvidia.com)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        b64_img = base64.b64encode(img_bytes).decode('utf-8')
        payload = {
            "input": [f"data:image/jpeg;base64,{b64_img}"]
        }
        try:
            response = requests.post(self.api_endpoint, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # Robust JSON parsing for NVIDIA NIM Hive Deepfake output
            fake_prob = 0.5
            data_list = result.get('data', [])
            if data_list:
                # Hive usually returns classes array
                classes = data_list[0].get('classes', [])
                for cls in classes:
                    if 'fake' in cls.get('name', '').lower() or 'deepfake' in cls.get('class', '').lower():
                        fake_prob = cls.get('score', 0.5)
                        break
            
            return {
                "score": float(fake_prob),
                "confidence": 0.95,
                "provider": self.provider
            }
        except Exception as e:
            print(f"NVIDIA NIM API Error: {e}")
            return self._mock_analyze(None, error=str(e))

    def _mock_analyze(self, image_pil, error=None):
        """
        Mock implementation for testing when no API key is available.
        It simulates a network delay and returns a neutral score.
        """
        # Simulate network latency
        time.sleep(0.5)
        return {
            "score": 0.5, # Neutral score
            "confidence": 0.1, # Low confidence because it's a mock
            "provider": "mock",
            "error": error
        }

if __name__ == "__main__":
    detector = ExternalAPIDetector()
    from PIL import Image
    import numpy as np
    dummy_img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    print(detector.analyze(dummy_img))
