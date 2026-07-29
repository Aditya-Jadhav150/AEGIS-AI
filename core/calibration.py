import numpy as np

class TemperatureScaler:
    """
    Calibrates XGBoost output probabilities using Temperature Scaling.
    """
    def __init__(self, temperature=1.5):
        self.temperature = temperature
        
    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """
        Applies temperature scaling to raw logits before softmax.
        """
        scaled_logits = logits / self.temperature
        # Compute softmax
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probs
