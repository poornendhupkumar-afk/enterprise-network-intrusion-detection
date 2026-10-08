#!/usr/bin/env python3
"""
tune_threshold.py
=================
Re-tunes the decision threshold and evaluates under custom asymmetric operational cost:
    Cost = cost_weight * FN + FP
"""

import sys
import joblib
import numpy as np
import pandas as pd
from features import FEATURE_NAMES, CATEGORICAL_COLS, NUMERIC_COLS, extract_features
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, average_precision_score

def evaluate(y_true, y_prob, threshold, cost_weight=40.0):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    cost = int(cost_weight * fn + fp)
    return {
        "threshold": threshold,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "cost": cost
    }

def optimize_threshold(y_true, y_prob, cost_weight=40.0, num_steps=1000):
    thresholds = np.linspace(0.0005, 0.90, num_steps)
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)

    best_thresh = 0.5
    lowest_cost = float("inf")
    for th in thresholds:
        pred_pos = (y_prob >= th)
        fn = np.count_nonzero(pos_mask & (~pred_pos))
        fp = np.count_nonzero(neg_mask & pred_pos)
        cost = cost_weight * fn + fp
        if cost < lowest_cost:
            lowest_cost = cost
            best_thresh = th

    best_metrics = evaluate(y_true, y_prob, threshold=best_thresh, cost_weight=cost_weight)
    return best_thresh, best_metrics

def main():
    cost_weight = 40.0
    if len(sys.argv) > 1:
        cost_weight = float(sys.argv[1])

    print("=" * 70)
    print(f" THRESHOLD OPTIMIZATION FOR COST WEIGHT = {cost_weight:.1f}")
    print(f" Cost Formula: Cost = {cost_weight:.0f} * FN + FP")
    print("=" * 70)

    model_path = "models/advanced_model_full.joblib"
    print(f"[*] Loading model artifact: {model_path} ...")
    package = joblib.load(model_path)
    cat_levels = package["cat_levels"]
    weights = package.get("weights", {"lgb": 0.60, "xgb": 0.40})

    val_path = "data/validation.csv"
    print(f"[*] Loading validation dataset: {val_path} ...")
    df_val = pd.read_csv(val_path, header=None, dtype=str, keep_default_na=False)
    y_val = pd.to_numeric(df_val[39], errors="coerce").fillna(0).astype(int).values
    attack_cats = df_val[38].astype(str).str.strip().replace({"": "Normal"}).values

    print("[*] Extracting features ...")
    val_features_df = df_val.iloc[:, :38]
    val_features_df.columns = FEATURE_NAMES
    X_val, _ = extract_features(val_features_df, cat_levels=cat_levels, is_train=False)

    print("[*] Generating model probabilities ...")
    p_lgb = package["model_lgb"].predict_proba(X_val)[:, 1]
    p_xgb = package["model_xgb"].predict_proba(X_val)[:, 1]
    y_prob = weights["lgb"] * p_lgb + weights["xgb"] * p_xgb

    # Evaluate at default threshold 0.50
    m_default = evaluate(y_val, y_prob, threshold=0.50, cost_weight=cost_weight)
    print("\n--- Standard Threshold (T = 0.50) ---")
    print(f"  F1: {m_default['f1']:.4f} | Prec: {m_default['precision']:.4f} | Rec: {m_default['recall']:.4f}")
    print(f"  Confusion: TP={m_default['tp']:,} | FP={m_default['fp']:,} | TN={m_default['tn']:,} | FN={m_default['fn']:,}")
    print(f"  Operational Cost: {m_default['cost']:,}  ({cost_weight:.0f}*FN + FP = {int(cost_weight * m_default['fn']):,} + {m_default['fp']:,})")

    # Evaluate at previous threshold 0.0276
    m_prev = evaluate(y_val, y_prob, threshold=0.0276, cost_weight=cost_weight)
    print(f"\n--- Previous Threshold (T = 0.0276) with Cost Weight {cost_weight:.0f} ---")
    print(f"  F1: {m_prev['f1']:.4f} | Prec: {m_prev['precision']:.4f} | Rec: {m_prev['recall']:.4f}")
    print(f"  Confusion: TP={m_prev['tp']:,} | FP={m_prev['fp']:,} | TN={m_prev['tn']:,} | FN={m_prev['fn']:,}")
    print(f"  Operational Cost: {m_prev['cost']:,}  ({cost_weight:.0f}*FN + FP = {int(cost_weight * m_prev['fn']):,} + {m_prev['fp']:,})")

    # Optimize threshold for cost_weight = 40
    best_thresh, m_opt = optimize_threshold(y_val, y_prob, cost_weight=cost_weight, num_steps=1000)
    cost_reduc = (1.0 - m_opt["cost"] / m_default["cost"]) * 100.0
    print(f"\n--- Re-Optimized Threshold (T* = {best_thresh:.4f}) for Cost Weight {cost_weight:.0f} ---")
    print(f"  F1: {m_opt['f1']:.4f} | Prec: {m_opt['precision']:.4f} | Rec: {m_opt['recall']:.4f}")
    print(f"  Confusion: TP={m_opt['tp']:,} | FP={m_opt['fp']:,} | TN={m_opt['tn']:,} | FN={m_opt['fn']:,}")
    print(f"  Operational Cost: {m_opt['cost']:,}  ({cost_reduc:.1f}% reduction vs default)")
    print(f"    Missed Intrusion Loss: {int(cost_weight * m_opt['fn']):,} ({m_opt['fn']:,} * {cost_weight:.0f})")
    print(f"    False Alarm Overhead:  {m_opt['fp']:,}")

    # Per category detection breakdown
    y_pred_opt = (y_prob >= best_thresh).astype(int)
    print("\nATTACK CATEGORY BREAKDOWN:")
    print(f"  {'Category':<18} {'Count':>8}   {'Metric':<20} {'Score':>8}")
    print("  " + "-" * 58)
    for cat in sorted(np.unique(attack_cats)):
        mask = (attack_cats == cat)
        cnt = int(np.sum(mask))
        cat_pred = y_pred_opt[mask]
        if cat == "Normal":
            spec = float(np.mean(cat_pred == 0))
            print(f"  {cat:<18} {cnt:>8,}   Specificity (Passed)   {spec:>7.2%}")
        else:
            rec = float(np.mean(cat_pred == 1))
            print(f"  {cat:<18} {cnt:>8,}   Recall (Detected)      {rec:>7.2%}")

    # Save updated package
    package["best_threshold"] = float(best_thresh)
    package["cost_weight"] = float(cost_weight)
    joblib.dump(package, "models/advanced_model_full.joblib")
    joblib.dump(package, "model.joblib")
    print("\n[+] Updated model artifacts saved with new threshold and cost_weight!")

    # Save predictions
    sub_df = pd.DataFrame({
        "row_id": np.arange(1, len(y_pred_opt) + 1),
        "prediction": y_pred_opt,
        "probability": np.round(y_prob, 6)
    })
    sub_df.to_csv("submissions/submission_val_cost40.csv", index=False)
    print(f"[+] Saved updated predictions to submissions/submission_val_cost40.csv")

if __name__ == "__main__":
    main()
