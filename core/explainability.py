import shap
import pandas as pd

class ExplainabilityEngine:
    """
    Generates SHAP explanations for XGBoost predictions.
    """
    def __init__(self, model):
        self.model = model
        self.explainer = shap.TreeExplainer(model)
        
    def generate_explanation(self, feature_df: pd.DataFrame) -> dict:
        """
        Returns feature importance contributions for a specific prediction.
        """
        shap_values = self.explainer.shap_values(feature_df)
        
        contributions = {}
        # shap_values could be a list if multi-class, or array if binary
        if isinstance(shap_values, list):
            sv = shap_values[1][0] # take positive class
        else:
            sv = shap_values[0]
            
        for i, col in enumerate(feature_df.columns):
            contributions[col] = float(sv[i])
            
        # Sort by absolute impact
        sorted_contrib = dict(sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True))
        
        return {
            "top_features": list(sorted_contrib.keys())[:3],
            "feature_contributions": sorted_contrib
        }
