import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def train_model(interval="1h"):
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"dataset_{interval}.csv")
    if not os.path.exists(dataset_path):
        print(f"[-] Dataset not found for {interval}. Please run dataset_builder.py {interval} first.")
        return

    print("[*] Loading dataset...")
    df = pd.read_csv(dataset_path)
    
    # Target is what we want to predict
    y = df['target']
    
    # Features (Drop the target)
    X = df.drop(columns=['target'])
    
    print(f"[*] Features: {list(X.columns)}")
    print(f"[*] Class distribution: \n{y.value_counts(normalize=True)}")

    # Split into Train and Test
    # For time series, it's better not to shuffle, but for this demo, standard split is okay.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    print("[*] Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    print("[*] Evaluating Model...")
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save Model
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"rf_model_{interval}.joblib")
    
    # Save the feature names as well so predictor knows what to feed in
    joblib.dump({"model": model, "features": list(X.columns)}, model_path)
    print(f"[+] Model saved successfully to {model_path}")

if __name__ == "__main__":
    import sys
    interval = sys.argv[1] if len(sys.argv) > 1 else "1h"
    train_model(interval)
