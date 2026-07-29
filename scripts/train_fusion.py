import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json

def train_xgboost(csv_path: str, output_model: str, output_scaler: str):
    """
    Trains the XGBoost fusion classifier on extracted metrics.
    """
    df = pd.read_csv(csv_path)
    
    # Assume 'label' column: 1 = fake, 0 = real
    if 'label' not in df.columns:
        raise ValueError("CSV must contain a 'label' column")
        
    X = df.drop(columns=['label', 'filename'], errors='ignore')
    y = df['label']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("Training Complete. Test Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    # Save model using JSON format
    model.save_model(output_model)
    
    # Save scaler in JSON format
    scaler_data = {
        "mean": scaler.mean_.tolist(),
        "var": scaler.var_.tolist(),
        "scale": scaler.scale_.tolist(),
        "n_features_in": scaler.n_features_in_
    }
    with open(output_scaler, 'w') as f:
        json.dump(scaler_data, f)
        
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python train_fusion.py path_to_features.csv")
    else:
        train_xgboost(sys.argv[1], "../dataset/fusion_engine_best.json", "../dataset/scaler.json")
