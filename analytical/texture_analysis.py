import numpy as np
import cv2

class TextureAnalyzer:
    def __init__(self):
        """
        Initializes the Texture Analyzer.
        """
        pass

    def _compute_lbp(self, img_gray):
        """
        Computes a simplified Local Binary Pattern (LBP) using numpy.
        """
        img = img_gray.astype(np.float32)
        out = np.zeros_like(img, dtype=np.uint8)
        
        # 8-neighbor LBP
        center = img[1:-1, 1:-1]
        out[1:-1, 1:-1] |= (img[0:-2, 0:-2] >= center).astype(np.uint8) * 1
        out[1:-1, 1:-1] |= (img[0:-2, 1:-1] >= center).astype(np.uint8) * 2
        out[1:-1, 1:-1] |= (img[0:-2, 2:] >= center).astype(np.uint8) * 4
        out[1:-1, 1:-1] |= (img[1:-1, 2:] >= center).astype(np.uint8) * 8
        out[1:-1, 1:-1] |= (img[2:, 2:] >= center).astype(np.uint8) * 16
        out[1:-1, 1:-1] |= (img[2:, 1:-1] >= center).astype(np.uint8) * 32
        out[1:-1, 1:-1] |= (img[2:, 0:-2] >= center).astype(np.uint8) * 64
        out[1:-1, 1:-1] |= (img[1:-1, 0:-2] >= center).astype(np.uint8) * 128
        
        return out

    def analyze(self, image_pil):
        """
        Analyzes the micro-textures of the image.
        Generators often struggle with micro-textures, leading to over-smoothing or unnatural patterns.
        :param image_pil: PIL Image.
        :return: dict with 'score' (0 to 1) and 'confidence'.
        """
        try:
            img_gray = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2GRAY)
            
            # Measure global blur/smoothness using Laplacian variance
            laplacian_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()
            
            # Deepfakes (especially early ones or strong face swaps) often have very low variance (over-smoothed skin)
            # Or very high variance (noise artifacts). We'll map this heuristically.
            # Normal well-lit face: 100 - 500
            smoothness_anomaly = 0.0
            if laplacian_var < 50.0: # Unnaturally smooth
                smoothness_anomaly = min(1.0, (50.0 - laplacian_var) / 50.0)
                
            # LBP Histogram analysis
            lbp = self._compute_lbp(img_gray)
            hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-7)
            
            # Calculate entropy of LBP histogram as a texture complexity measure
            entropy = -np.sum(hist * np.log2(hist + 1e-7))
            
            # Normal face texture entropy is usually around 5.5 - 7.5 depending on resolution
            # Very low entropy means lack of texture variation (typical of AI smoothing)
            texture_anomaly = 0.0
            if entropy < 5.0:
                texture_anomaly = min(1.0, (5.0 - entropy) / 2.0)
            elif entropy > 7.8: # Unnatural noise
                texture_anomaly = min(1.0, (entropy - 7.8) / 1.0)
                
            # Combine signals
            final_score = (smoothness_anomaly * 0.4) + (texture_anomaly * 0.6)
            
            return {
                "score": float(final_score),
                "confidence": 0.6,
                "anomaly_detected": final_score > 0.4,
                "laplacian_variance": float(laplacian_var),
                "lbp_entropy": float(entropy)
            }
            
        except Exception as e:
            print(f"Texture Analysis Error: {e}")
            return {"score": 0.0, "confidence": 0.0, "anomaly_detected": False, "error": str(e)}

if __name__ == "__main__":
    from PIL import Image
    import sys
    if len(sys.argv) > 1:
        img = Image.open(sys.argv[1]).convert("RGB")
        analyzer = TextureAnalyzer()
        res = analyzer.analyze(img)
        print(f"Texture Result: {res}")
