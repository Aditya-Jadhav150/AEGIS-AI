import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib

DATA_CSV = os.path.join('dataset', 'fusion_features.csv')

def main():
    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(f"Feature CSV not found at {DATA_CSV}. Run extract_fusion_features.py first.")

    df = pd.read_csv(DATA_CSV)
    X = df[[
        'spatial_score', 'freq_score', 'latent_score', 'stat_score',
        'entropy', 'edge_density', 'laplacian_variance', 'color_kurtosis', 'jpeg_consistency'
    ]]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train a lightweight XGBoost classifier
    # Fit a StandardScaler on the training features for consistent inference
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        n_jobs=4,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # Predictions & evaluation (using scaled features)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n=== Fusion Engine Evaluation ===")
    print(f"Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['real', 'fake']))

    # Feature importances (explainability)
    importance = model.get_booster().get_score(importance_type='weight')
    print("\nFeature Importances (higher = more important):")
    for feat, score in sorted(importance.items(), key=lambda kv: kv[1], reverse=True):
        print(f"{feat}: {score}")

    # Show a few example rows with their predicted probabilities
    probs = model.predict_proba(X_test_scaled)[:, 1]
    example_df = pd.DataFrame(X_test, columns=X.columns)
    example_df['true_label'] = y_test.values
    example_df['pred_prob_fake'] = probs
    print("\nSample predictions (first 5 rows):")
    print(example_df.head(5))

    # --------------------------------------------------------------
    # Persist the trained model and scaler for inference (Plain Text JSON)
    # --------------------------------------------------------------
    import json
    model_json_path = os.path.join('dataset', 'fusion_engine_best.json')
    scaler_json_path = os.path.join('dataset', 'scaler.json')
    
    # Save XGBoost model to JSON
    model.save_model(model_json_path)
    
    # Save StandardScaler parameters to JSON
    scaler_data = {
        "mean": scaler.mean_.tolist(),
        "var": scaler.var_.tolist(),
        "scale": scaler.scale_.tolist(),
        "n_features_in": int(scaler.n_features_in_)
    }
    with open(scaler_json_path, 'w') as f:
        json.dump(scaler_data, f)
        
    print(f"🗄️  Model saved as text to {model_json_path}")
    print(f"📏  Scaler saved as text to {scaler_json_path}")

if __name__ == '__main__':
    main()
