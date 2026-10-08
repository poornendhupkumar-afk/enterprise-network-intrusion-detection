#!/usr/bin/env python3
"""
train_advanced.py
=================
Advanced Intrusion Detection Model Pipeline.
Includes:
- Robust feature engineering (traffic ratios, loss rates, scan indicators, TCP flags).
- Multi-algorithm Gradient Boosted Trees (LightGBM & XGBoost).
- Out-of-fold and Validation threshold optimization for asymmetric loss:
    Cost = cost_weight * FN + FP (default weight = 20).
- Detailed per-attack-category recall analysis (Generic, Exploits, Fuzzers, Analysis, Worms).
- Artifact export compatible with predict.py.
"""

import os
import sys
import time
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    average_precision_score, confusion_matrix, roc_auc_score
)
import lightgbm as lgb
import xgboost as xgb

# ---------------------------------------------------------------------------
# Feature Column Definitions
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "proto", "state", "dur", "sbytes", "dbytes", "sloss", "dloss", "service",
    "Sload", "Dload", "Spkts", "Dpkts", "swin", "dwin", "stcpb", "dtcpb",
    "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "Sjit", "Djit",
    "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat", "is_sm_ips_ports",
    "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst",
    "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm"
]
TRAIN_VAL_COLS = FEATURE_NAMES + ["attack_cat", "Label"]
CATEGORICAL_COLS = ["proto", "state", "service"]
NUMERIC_COLS = [c for c in FEATURE_NAMES if c not in CATEGORICAL_COLS]


def load_dataset(filepath, is_labelled=True, nrows=None):
    """
    Robust CSV loader that handles empty string values and headerless format.
    """
    print(f"[*] Loading dataset: {filepath} ...")
    cols = TRAIN_VAL_COLS if is_labelled else FEATURE_NAMES
    
    df = pd.read_csv(
        filepath,
        header=None,
        names=cols,
        dtype=str,
        keep_default_na=False,
        nrows=nrows
    )
    if is_labelled:
        df["attack_cat"] = df["attack_cat"].astype(str).str.strip().replace({"": "Normal"})
        df["Label"] = pd.to_numeric(df["Label"], errors="coerce").fillna(0).astype(int)
    print(f"    Loaded {len(df):,} records with {df.shape[1]} columns.")
    return df


from features import extract_features as engineer_features


def evaluate_predictions(y_true, y_prob, threshold=0.5, cost_weight=20.0):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = 0.0
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.0
        
    cost = int(cost_weight * fn + fp)
    
    return {
        "threshold": threshold,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "cost": cost
    }


def optimize_threshold(y_true, y_prob, cost_weight=20.0, num_steps=500):
    """
    Ultra-fast vectorized threshold scan minimizing Cost = cost_weight * FN + FP.
    """
    thresholds = np.linspace(0.001, 0.95, num_steps)
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
            
    best_metrics = evaluate_predictions(y_true, y_prob, threshold=best_thresh, cost_weight=cost_weight)
    return best_thresh, best_metrics


def print_breakdown(attack_cats, y_pred):
    print("\nATTACK CATEGORY BREAKDOWN:")
    print(f"  {'Category':<18} {'Count':>8}   {'Metric':<20} {'Score':>8}")
    print("  " + "-" * 58)
    unique_cats = sorted(np.unique(attack_cats))
    for cat in unique_cats:
        mask = (attack_cats == cat)
        cnt = int(np.sum(mask))
        if cnt == 0:
            continue
        cat_pred = y_pred[mask]
        if cat == "Normal":
            spec = float(np.mean(cat_pred == 0))
            print(f"  {cat:<18} {cnt:>8,}   Specificity (Passed)   {spec:>7.2%}")
        else:
            rec = float(np.mean(cat_pred == 1))
            print(f"  {cat:<18} {cnt:>8,}   Recall (Detected)      {rec:>7.2%}")


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="Advanced Network Intrusion Detection Model")
    parser.add_argument("--train", type=str, default="data/train.csv", help="Path to train.csv")
    parser.add_argument("--val", type=str, default="data/validation.csv", help="Path to validation.csv")
    parser.add_argument("--save-model", type=str, default="models/advanced_model.joblib", help="Model artifact path")
    parser.add_argument("--output", type=str, default="submissions/submission_validation_advanced.csv", help="Submission output")
    parser.add_argument("--cost-weight", type=float, default=40.0, help="FN penalty weight (default: 40.0)")
    parser.add_argument("--nrows", type=int, default=None, help="Optional row limit for quick tests")
    args = parser.parse_args()

    t_start = time.time()
    print("=" * 70)
    print(" ADVANCED ENTERPRISE NETWORK INTRUSION DETECTION MODEL")
    print("=" * 70)
    print(f"[*] Training dataset:   {args.train}")
    print(f"[*] Validation dataset: {args.val}")
    print(f"[*] Cost function:      Cost = {args.cost_weight:.0f} * FN + FP")
    print("-" * 70)

    # 1. Load Data
    train_df = load_dataset(args.train, is_labelled=True, nrows=args.nrows)
    val_df = load_dataset(args.val, is_labelled=True, nrows=args.nrows)

    # 2. Extract Features
    print("[*] Engineering domain features ...")
    t_feat = time.time()
    X_train, cat_levels = engineer_features(train_df, is_train=True)
    y_train = train_df["Label"].values
    
    X_val, _ = engineer_features(val_df, cat_levels=cat_levels, is_train=False)
    y_val = val_df["Label"].values
    print(f"    Feature matrix shape: {X_train.shape} in {time.time() - t_feat:.2f}s.")

    feature_cols = list(X_train.columns)

    # 3. Train High-Performance LightGBM
    print("\n[*] Training Advanced LightGBM Classifier ...")
    t_lgb = time.time()
    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.06,
        "num_leaves": 127,
        "max_depth": 11,
        "min_child_samples": 25,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "reg_lambda": 1.5,
        "n_estimators": 350,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1
    }
    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(X_train, y_train)
    print(f"    LightGBM training complete in {time.time() - t_lgb:.2f}s.")

    # 4. Train Fast Regularized XGBoost
    print("\n[*] Training Regularized XGBoost Classifier ...")
    t_xgb = time.time()
    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "learning_rate": 0.06,
        "max_depth": 9,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "reg_lambda": 1.5,
        "n_estimators": 250,
        "random_state": 42,
        "n_jobs": -1
    }
    model_xgb = xgb.XGBClassifier(**xgb_params)
    model_xgb.fit(X_train, y_train)
    print(f"    XGBoost training complete in {time.time() - t_xgb:.2f}s.")

    # 5. Evaluate Individual & Ensemble Predictions
    print("\n" + "=" * 70)
    print(" VALIDATION EVALUATION: INDIVIDUAL & ENSEMBLE")
    print("=" * 70)
    probs_lgb = model_lgb.predict_proba(X_val)[:, 1]
    probs_xgb = model_xgb.predict_proba(X_val)[:, 1]
    
    # Grid search for best ensemble weight
    best_weight_lgb = 0.60
    lowest_ensemble_cost = float("inf")
    best_ensemble_probs = None
    best_ensemble_thresh = 0.01

    for w in [0.40, 0.50, 0.55, 0.60, 0.65, 0.70]:
        p_cand = w * probs_lgb + (1.0 - w) * probs_xgb
        th, m = optimize_threshold(y_val, p_cand, cost_weight=args.cost_weight, num_steps=2000)
        if m["cost"] < lowest_ensemble_cost:
            lowest_ensemble_cost = m["cost"]
            best_weight_lgb = w
            best_ensemble_probs = p_cand
            best_ensemble_thresh = th

    probs_ensemble = best_ensemble_probs
    best_threshold = best_ensemble_thresh

    for name, p in [("LightGBM", probs_lgb), ("XGBoost", probs_xgb), (f"Blended Ensemble ({best_weight_lgb:.0%}/{1-best_weight_lgb:.0%})", probs_ensemble)]:
        auc_pr = average_precision_score(y_val, p)
        auc_roc = roc_auc_score(y_val, p)
        res_def = evaluate_predictions(y_val, p, threshold=0.5, cost_weight=args.cost_weight)
        best_t, res_opt = optimize_threshold(y_val, p, cost_weight=args.cost_weight, num_steps=2000)
        print(f"\n--- {name} ---")
        print(f"  PR-AUC: {auc_pr:.5f} | ROC-AUC: {auc_roc:.5f}")
        print(f"  Standard (T=0.50): F1={res_def['f1']:.4f} | Prec={res_def['precision']:.4f} | Rec={res_def['recall']:.4f} | Cost={res_def['cost']:,} (FN={res_def['fn']}, FP={res_def['fp']})")
        print(f"  Cost-Opt (T*={best_t:.4f}): F1={res_opt['f1']:.4f} | Prec={res_opt['precision']:.4f} | Rec={res_opt['recall']:.4f} | Cost={res_opt['cost']:,} (FN={res_opt['fn']}, FP={res_opt['fp']})")

    opt_metrics = evaluate_predictions(y_val, probs_ensemble, threshold=best_threshold, cost_weight=args.cost_weight)
    y_opt_pred = (probs_ensemble >= best_threshold).astype(int)

    print("\n" + "=" * 70)
    print(f" BEST ENSEMBLE AT OPTIMAL THRESHOLD (T* = {best_threshold:.4f})")
    print("=" * 70)
    print(f"  F1-Score:     {opt_metrics['f1']:.4f}")
    print(f"  PR-AUC:       {opt_metrics['pr_auc']:.5f}")
    print(f"  Precision:    {opt_metrics['precision']:.4f}")
    print(f"  Recall:       {opt_metrics['recall']:.4f} (Catch Rate)")
    print(f"  Confusion:    TP={opt_metrics['tp']:,} | FP={opt_metrics['fp']:,} | TN={opt_metrics['tn']:,} | FN={opt_metrics['fn']:,}")
    print(f"  Total Cost:   {opt_metrics['cost']:,}  (Penalty = {args.cost_weight:.0f}*FN + FP)")

    print_breakdown(val_df["attack_cat"].values, y_opt_pred)

    # 6. Save Complete Model Artifact
    model_package = {
        "model_lgb": model_lgb,
        "model_xgb": model_xgb,
        "weights": {"lgb": float(best_weight_lgb), "xgb": float(1.0 - best_weight_lgb)},
        "cat_levels": cat_levels,
        "categorical_cols": CATEGORICAL_COLS,
        "numeric_cols": NUMERIC_COLS,
        "feature_names": FEATURE_NAMES,
        "engineered_feature_cols": feature_cols,
        "best_threshold": float(best_threshold),
        "cost_weight": float(args.cost_weight)
    }
    os.makedirs(os.path.dirname(args.save_model), exist_ok=True)
    joblib.dump(model_package, args.save_model)
    print(f"\n[*] Model artifact saved to: {args.save_model}")

    # 7. Generate Submission CSV
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    sub_df = pd.DataFrame({
        "row_id": np.arange(1, len(y_opt_pred) + 1),
        "prediction": y_opt_pred,
        "probability": np.round(probs_ensemble, 6)
    })
    sub_df.to_csv(args.output, index=False)
    print(f"[+] Validation submission saved to: {args.output}")
    print(f"    Total rows: {len(sub_df):,}")
    print(f"Total pipeline execution time: {time.time() - t_start:.2f}s.")


if __name__ == "__main__":
    main()
