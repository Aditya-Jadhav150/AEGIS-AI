import numpy as np
from facenet_pytorch import MTCNN
import torch
import cv2

class StructureAnalyzer:
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.mtcnn = MTCNN(keep_all=False, device=self.device)

    def analyze(self, image_pil):
        """
        Analyzes facial structure symmetry and alignment using MTCNN landmarks.
        :param image_pil: PIL Image.
        :return: dict with 'score' (0 to 1) and 'confidence'.
        """
        try:
            boxes, probs, landmarks = self.mtcnn.detect(image_pil, landmarks=True)
            
            if landmarks is None or len(landmarks) == 0:
                return {"score": 0.0, "confidence": 0.0, "anomaly_detected": False, "error": "No face/landmarks detected"}
                
            # Take the highest probability face
            lm = landmarks[0]
            
            # MTCNN landmarks: [left_eye, right_eye, nose, left_mouth, right_mouth]
            left_eye = lm[0]
            right_eye = lm[1]
            nose = lm[2]
            left_mouth = lm[3]
            right_mouth = lm[4]
            
            # Structural Heuristics:
            # 1. Eye symmetry: Y-coordinate difference should be relatively small
            # (Though tilting the head changes this, we normalize by eye distance)
            eye_dist = np.linalg.norm(left_eye - right_eye) + 1e-7
            eye_y_diff = abs(left_eye[1] - right_eye[1]) / eye_dist
            
            # 2. Nose centering: The horizontal distance from nose to left eye vs right eye
            dist_nose_left = np.linalg.norm(nose - left_eye)
            dist_nose_right = np.linalg.norm(nose - right_eye)
            nose_symmetry = abs(dist_nose_left - dist_nose_right) / (eye_dist + 1e-7)
            
            # 3. Mouth centering
            mouth_center = (left_mouth + right_mouth) / 2
            mouth_nose_x_diff = abs(mouth_center[0] - nose[0]) / (eye_dist + 1e-7)
            
            # Compute anomaly score based on heuristics
            # If the face is highly asymmetrical, it could be an artifact of poor generation.
            # However, extreme poses also cause this, so we must be careful.
            score = 0.0
            
            # If eye y-diff is very high but the head isn't tilted (which is hard to know without full 3d pose)
            # We'll just combine these weakly.
            anomaly_factor = eye_y_diff + nose_symmetry + mouth_nose_x_diff
            
            # Typical faces have anomaly_factor < 0.5 depending on pose
            if anomaly_factor > 0.8:
                score = min(1.0, (anomaly_factor - 0.8) * 2.0)
                
            return {
                "score": float(score),
                "confidence": 0.4, # Structural analysis alone from 5 points is weak
                "anomaly_detected": score > 0.5,
                "metrics": {
                    "eye_y_diff_norm": float(eye_y_diff),
                    "nose_symmetry_norm": float(nose_symmetry),
                    "mouth_nose_x_diff_norm": float(mouth_nose_x_diff)
                }
            }
            
        except Exception as e:
            print(f"Structural Analysis Error: {e}")
            return {"score": 0.0, "confidence": 0.0, "anomaly_detected": False, "error": str(e)}

if __name__ == "__main__":
    from PIL import Image
    import sys
    if len(sys.argv) > 1:
        img = Image.open(sys.argv[1]).convert("RGB")
        analyzer = StructureAnalyzer()
        res = analyzer.analyze(img)
        print(f"Structure Result: {res}")
