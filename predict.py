#!/usr/bin/env python3
"""
predict.py
==========
ML Hackathon - Network Intrusion Detection
Universal Model Inference Script

Complies with the official competition requirements:
    python predict.py --input path/to/test_features.csv --output predictions.csv

Supports:
- Advanced ensemble models (LightGBM + XGBoost + Feature Engineering)
- Standard baseline models (HistGradientBoosting)
- Custom decision thresholds and automatic cost-optimal threshold application
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd

from features import FEATURE_NAMES, CATEGORICAL_COLS, NUMERIC_COLS, extract_features


def preprocess_baseline(df, cat_levels):
    """Fallback preprocessor for single baseline models without engineered features."""
    X = pd.DataFrame(index=df.index)
    for c in CATEGORICAL_COLS:
        mapping = cat_levels.get(c, {})
        oov_idx = len(mapping)
        X[c] = df[c].astype(str).map(mapping).fillna(oov_idx).astype(int)
    for c in NUMERIC_COLS:
        X[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(np.float32)
    return X


def resolve_model_path(model_path):
    candidates = [
        model_path,
        os.path.join(os.path.dirname(__file__), model_path),
        os.path.join(os.path.dirname(__file__), "models", "advanced_model.joblib"),
        os.path.join(os.path.dirname(__file__), "models", "baseline_full.joblib"),
        "models/advanced_model.joblib",
        "models/baseline_full.joblib",
        "baseline_model.joblib"
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return model_path


def main():
    parser = argparse.ArgumentParser(description="Network Intrusion Detection Inference")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input unlabelled CSV (38 feature columns)")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to save output prediction CSV")
    parser.add_argument("--model", "-m", type=str, default="models/advanced_model_full.joblib", help="Path to saved model artifact")
    parser.add_argument("--threshold", "-t", type=float, default=None, help="Optional probability decision threshold override")
    args = parser.parse_args()

    model_path = resolve_model_path(args.model)
    if not os.path.exists(model_path):
        print(f"[!] Error: Model artifact not found: {args.model}")
        print("    Tried looking in default candidate paths.")
        sys.exit(1)

    print(f"[*] Loading model artifact: {model_path} ...")
    package = joblib.load(model_path)
    cat_levels = package.get("cat_levels", {})
    threshold = args.threshold if args.threshold is not None else package.get("best_threshold", 0.5)

    print(f"[*] Reading test features: {args.input} ...")
    test_df = pd.read_csv(
        args.input,
        header=None,
        usecols=list(range(len(FEATURE_NAMES))),
        names=FEATURE_NAMES,
        dtype=str,
        keep_default_na=False
    )
    print(f"    Loaded {len(test_df):,} records.")

    # Determine model type and feature pipeline
    has_ensemble = "model_lgb" in package and "model_xgb" in package
    has_engineered = "engineered_feature_cols" in package

    if has_engineered:
        print("[*] Running domain feature engineering ...")
        X_test, _ = extract_features(test_df, cat_levels=cat_levels, is_train=False)
    else:
        print("[*] Running standard baseline preprocessing ...")
        X_test = preprocess_baseline(test_df, cat_levels)

    print(f"[*] Generating model predictions (Decision Threshold = {threshold:.4f}) ...")
    if has_ensemble:
        model_lgb = package["model_lgb"]
        model_xgb = package["model_xgb"]
        weights = package.get("weights", {"lgb": 0.60, "xgb": 0.40})
        p_lgb = model_lgb.predict_proba(X_test)[:, 1]
        p_xgb = model_xgb.predict_proba(X_test)[:, 1]
        probs = weights["lgb"] * p_lgb + weights["xgb"] * p_xgb
    elif "model" in package:
        probs = package["model"].predict_proba(X_test)[:, 1]
    else:
        raise ValueError("Invalid model package: Neither 'model' nor ensemble found in artifact.")

    preds = (probs >= threshold).astype(int)

    # Format submission DataFrame
    sub_df = pd.DataFrame({
        "row_id": np.arange(1, len(preds) + 1),
        "prediction": preds,
        "probability": np.round(probs, 6)
    })

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    sub_df.to_csv(args.output, index=False)
    print(f"[+] Predictions successfully saved to: {args.output}")
    print(f"    Total predictions: {len(sub_df):,}")
    print(f"    Predicted Attacks: {int(np.sum(preds == 1)):,} ({np.mean(preds == 1):.2%})")
    print(f"    Predicted Normal:  {int(np.sum(preds == 0)):,} ({np.mean(preds == 0):.2%})")


if __name__ == "__main__":
    main()
