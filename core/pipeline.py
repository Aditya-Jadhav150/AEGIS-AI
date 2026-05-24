import torch
from PIL import Image, ImageOps
from facenet_pytorch import MTCNN
import concurrent.futures
import json
import time
import sys
import os

# Add the root directory to sys.path to allow imports when running as a script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
from detectors.external_api import ExternalAPIDetector
from detectors.custom_model import CustomModelDetector
from analytical.fft_analysis import FFTAnalyzer
from analytical.texture_analysis import TextureAnalyzer
from analytical.structure import StructureAnalyzer
from core.fusion import FusionEngine

class HybridDeepfakeDetector:
    def __init__(self, api_endpoint=None, api_key=None, provider="mock", device=None):
        """
        Initializes the entire detection pipeline.
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print("Initializing Hybrid Deepfake Detection Pipeline...")
        print(f"Target Device: {self.device}")
        
        # 1. Face Extractor
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        
        # 2. Detection Layers
        self.api_detector = ExternalAPIDetector(api_endpoint, api_key, provider)
        self.custom_detector = CustomModelDetector(device=self.device)
        
        # 3. Analytical Validation
        self.fft_analyzer = FFTAnalyzer()
        self.texture_analyzer = TextureAnalyzer()
        self.structure_analyzer = StructureAnalyzer(device=self.device)
        
        # 4. Fusion Engine
        self.fusion_engine = FusionEngine()
        
        print("Pipeline initialization complete.\n")

    def analyze_image(self, image_path):
        """
        Runs the full hybrid detection pipeline on an image.
        :param image_path: Path to the image file.
        :return: JSON string of the results.
        """
        try:
            # 1. Input Handling & Normalization
            # Read EXIF orientation natively to ensure correct upright image
            image = Image.open(image_path)
            image = ImageOps.exif_transpose(image).convert("RGB")
            
            # Extract faces
            boxes, probs = self.mtcnn.detect(image)
            
            if boxes is None or len(boxes) == 0:
                # No faces detected, run full image analysis as a fallback?
                # The requirements say "focused on human faces". We return early.
                res = self.fusion_engine.aggregate_image([])
                return json.dumps(res, indent=4)
            
            face_results = []
            
            # We process each face
            for i, box in enumerate(boxes):
                if probs[i] < 0.90:
                    continue # Skip low-confidence face detections
                    
                # Expand box exactly by 15% to grab jawline and hairline (deepfake seams)
                w, h = box[2] - box[0], box[3] - box[1]
                b1, b2 = max(0, box[0] - w * 0.15), max(0, box[1] - h * 0.15)
                b3, b4 = min(image.width, box[2] + w * 0.15), min(image.height, box[3] + h * 0.15)
                
                face_crop = image.crop((int(b1), int(b2), int(b3), int(b4)))
                bbox_list = [int(b1), int(b2), int(w), int(h)]
                
                # Execute modules concurrently to optimize performance
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    future_api = executor.submit(self.api_detector.analyze, face_crop)
                    future_custom = executor.submit(self.custom_detector.analyze, face_crop)
                    future_fft = executor.submit(self.fft_analyzer.analyze, face_crop)
                    future_texture = executor.submit(self.texture_analyzer.analyze, face_crop)
                    
                    # Structure analyzer needs the original full image and bounding box context
                    # Or we just pass the crop and let it find the face inside the crop
                    future_structure = executor.submit(self.structure_analyzer.analyze, face_crop)
                    
                    api_res = future_api.result()
                    custom_res = future_custom.result()
                    fft_res = future_fft.result()
                    texture_res = future_texture.result()
                    structure_res = future_structure.result()
                
                # Fuse Results
                fused_face = self.fusion_engine.fuse(
                    face_bbox=bbox_list,
                    external_res=api_res,
                    custom_res=custom_res,
                    fft_res=fft_res,
                    texture_res=texture_res,
                    structure_res=structure_res
                )
                
                face_results.append(fused_face)
                
            # Aggregate to Image Level
            final_report = self.fusion_engine.aggregate_image(face_results)
            
            return json.dumps(final_report, indent=4)
            
        except Exception as e:
            return json.dumps({
                "overall_verdict": "Unknown",
                "confidence_score": 0.0,
                "faces": [],
                "explanation": [f"Pipeline error: {str(e)}"]
            }, indent=4)

if __name__ == "__main__":
    import sys
    import os
    
    detector = HybridDeepfakeDetector()
    
    test_img = "test.jpg"
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        test_img = sys.argv[1]
        
    if os.path.exists(test_img):
        print(f"Analyzing {test_img}...\n")
        start_time = time.time()
        result_json = detector.analyze_image(test_img)
        end_time = time.time()
        print(result_json)
        print(f"\nProcessing Time: {end_time - start_time:.2f} seconds")
    else:
        print(f"Test image {test_img} not found.")
