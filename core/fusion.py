class FusionEngine:
    def __init__(self):
        """
        Initializes the Fusion Engine with predefined weights.
        """
        # Define weights for each module's contribution to the final score
        self.weights = {
            "external_api": 0.40,
            "custom_model": 0.40,
            "fft_analysis": 0.10,
            "texture_analysis": 0.05,
            "structure_analysis": 0.05
        }
        
        # Decision thresholds
        self.thresholds = {
            "fake": 0.65,
            "suspicious": 0.40
        }

    def fuse(self, face_bbox, external_res, custom_res, fft_res, texture_res, structure_res):
        """
        Fuses the results from all modules for a single face.
        :return: dict containing the fused classification, score, and explanation.
        """
        explanations = []
        
        # Extract fake probabilities (scores are expected to be prob of being FAKE)
        api_score = external_res.get("score", 0.5)
        custom_score = custom_res.get("score", 0.5)
        fft_score = fft_res.get("score", 0.0)
        texture_score = texture_res.get("score", 0.0)
        structure_score = structure_res.get("score", 0.0)
        
        # Calculate weighted final score
        # Note: If a module failed (confidence == 0), we could dynamically re-weight.
        # For simplicity, we assume they all run, or return a neutral 0.5/0.0 score.
        final_score = (
            (api_score * self.weights["external_api"]) +
            (custom_score * self.weights["custom_model"]) +
            (fft_score * self.weights["fft_analysis"]) +
            (texture_score * self.weights["texture_analysis"]) +
            (structure_score * self.weights["structure_analysis"])
        )
        
        # Generate Explanations based on individual thresholds
        if api_score > 0.6:
            explanations.append(f"High API confidence ({api_score*100:.1f}%)")
        elif api_score < 0.4:
            explanations.append(f"Low API confidence ({api_score*100:.1f}%)")
            
        if custom_score > 0.6:
            explanations.append(f"High Custom Model confidence ({custom_score*100:.1f}%)")
            
        if fft_score > 0.5:
            explanations.append("Frequency anomaly detected (FFT)")
            
        if texture_score > 0.4:
            explanations.append("Texture inconsistency observed (LBP/Laplacian)")
            
        if structure_score > 0.5:
            explanations.append("Structural anomaly detected (Symmetry/Alignment)")

        if len(explanations) == 0:
            explanations.append("No significant anomalies detected across layers.")

        # Decision Logic
        if final_score >= self.thresholds["fake"]:
            classification = "Fake"
        elif final_score >= self.thresholds["suspicious"]:
            classification = "Suspicious"
        else:
            classification = "Real"

        # Check for conflicts (e.g. API says 90% fake, custom says 10% fake)
        if abs(api_score - custom_score) > 0.5:
            classification = "Suspicious"
            explanations.append("CONFLICT: Primary and Secondary models strongly disagree.")

        return {
            "bbox": face_bbox,
            "classification": classification,
            "confidence": float(final_score),
            "explanation": explanations,
            "raw_scores": {
                "api": float(api_score),
                "custom": float(custom_score),
                "fft": float(fft_score),
                "texture": float(texture_score),
                "structure": float(structure_score)
            }
        }

    def aggregate_image(self, face_results):
        """
        Aggregates per-face results into an overall image verdict.
        If ANY face is Fake, the image is Fake. If ANY face is Suspicious, the image is Suspicious.
        """
        if not face_results:
            return {
                "overall_verdict": "Unknown",
                "confidence_score": 0.0,
                "faces": [],
                "explanation": ["No faces detected in the image."]
            }

        max_score = 0.0
        overall_classification = "Real"
        image_explanations = set()

        for res in face_results:
            cls = res["classification"]
            score = res["confidence"]
            
            if score > max_score:
                max_score = score
                
            if cls == "Fake":
                overall_classification = "Fake"
            elif cls == "Suspicious" and overall_classification == "Real":
                overall_classification = "Suspicious"
                
            for exp in res["explanation"]:
                image_explanations.add(exp)

        # Multi-face consistency check
        classes_found = set(r["classification"] for r in face_results)
        if "Real" in classes_found and "Fake" in classes_found:
            image_explanations.add("MULTI-FACE CONFLICT: Both highly confident Real and Fake faces found in the same image.")
            overall_classification = "Suspicious"

        return {
            "overall_verdict": overall_classification,
            "confidence_score": float(max_score),
            "faces": face_results,
            "explanation": list(image_explanations)
        }
