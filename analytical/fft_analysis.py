import numpy as np
import cv2
from scipy import ndimage

class FFTAnalyzer:
    def __init__(self, high_freq_threshold=0.65):
        """
        Initializes the FFT Analyzer.
        :param high_freq_threshold: The percentile threshold to consider as high frequency.
        """
        self.high_freq_threshold = high_freq_threshold

    def analyze(self, image_pil):
        """
        Analyzes the image in the frequency domain.
        :param image_pil: PIL Image.
        :return: dict with 'anomaly_score' (0 to 1) and 'confidence'.
        """
        try:
            # Convert to grayscale numpy array
            img_gray = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2GRAY)
            
            # Compute 2D FFT
            f = np.fft.fft2(img_gray)
            fshift = np.fft.fftshift(f)
            
            # Compute magnitude spectrum
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
            
            # Calculate radial profile (azimuthal average)
            h, w = magnitude_spectrum.shape
            y, x = np.indices((h, w))
            center = (h // 2, w // 2)
            r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
            r = r.astype(int)
            
            tbin = np.bincount(r.ravel(), magnitude_spectrum.ravel())
            nr = np.bincount(r.ravel())
            radialprofile = tbin / np.maximum(nr, 1)
            
            # Analyze high frequency components
            # GANs often have artifacts in the higher frequency bins
            total_bins = len(radialprofile)
            high_freq_start = int(total_bins * self.high_freq_threshold)
            
            if high_freq_start >= total_bins:
                return {"score": 0.0, "confidence": 0.0, "anomaly_detected": False}
                
            high_freq_energy = np.mean(radialprofile[high_freq_start:])
            low_freq_energy = np.mean(radialprofile[:high_freq_start])
            
            # Calculate a ratio. Real images usually have smoothly decaying high frequencies.
            # An unusually high ratio might indicate upsampling/GAN artifacts.
            ratio = high_freq_energy / (low_freq_energy + 1e-8)
            
            # Normalize to an anomaly score between 0 and 1 (heuristically tuned)
            # Typically ratio is very low. High-res real images can spike to 0.3 naturally.
            # Deepfakes (GANs) often push this ratio above 0.8
            score = 0.0
            if ratio > 0.4:
                score = min(1.0, (ratio - 0.4) * 2.5) 
            
            return {
                "score": float(score),
                "confidence": 0.5, # Reduced confidence as FFT is highly variable
                "anomaly_detected": score > 0.5,
                "ratio": float(ratio)
            }
            
        except Exception as e:
            print(f"FFT Analysis Error: {e}")
            return {"score": 0.0, "confidence": 0.0, "anomaly_detected": False, "error": str(e)}

if __name__ == "__main__":
    # Test script
    from PIL import Image
    import sys
    if len(sys.argv) > 1:
        img = Image.open(sys.argv[1]).convert("RGB")
        analyzer = FFTAnalyzer()
        res = analyzer.analyze(img)
        print(f"FFT Result: {res}")
