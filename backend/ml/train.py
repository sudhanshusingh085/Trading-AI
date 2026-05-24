import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, make_scorer, fbeta_score

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
    print(f"[*] Total dataset size: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"[*] Class distribution: \n{y.value_counts(normalize=True)}")

    # TimeSeriesSplit prevents sequential leakage
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Define hyperparameter grid for tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [6, 8, 10],
        'min_samples_leaf': [3, 5, 10],
        'max_features': ['sqrt', 0.5]
    }
    
    # Define custom F0.5 scorer (weights Precision twice as high as Recall to eliminate false positives)
    f05_scorer = make_scorer(fbeta_score, beta=0.5)
    
    print("[*] Performing Time-Series Cross-Validation Grid Search...")
    base_model = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=tscv,
        scoring=f05_scorer,
        verbose=1,
        n_jobs=-1
    )
    
    # Standard time series split for final evaluation
    # Use the last 20% of chronological data as lockbox test set
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"[+] Best Parameters: {grid_search.best_params_}")
    print(f"[+] Best CV F0.5 Score: {grid_search.best_score_:.4f}")

    print("[*] Evaluating Best Model on Lockbox Out-of-Sample (OOS) Test Set...")
    y_pred = best_model.predict(X_test)
    
    print(f"OOS Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"OOS Precision (Profitable Trades): {precision_score(y_test, y_pred):.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save Model
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"rf_model_{interval}.joblib")
    
    # Save the feature names as well so predictor knows what to feed in
    joblib.dump({"model": best_model, "features": list(X.columns)}, model_path)
    print(f"[+] Highly-optimized Model saved successfully to {model_path}")

if __name__ == "__main__":
    import sys
    interval = sys.argv[1] if len(sys.argv) > 1 else "1h"
    train_model(interval)
