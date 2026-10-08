#!/usr/bin/env python3
"""
eda.py
======
Exploratory Data Analysis for UNSW-NB15 Hackathon Dataset.
Analyzes class distribution, attack categories, feature distributions, missing values, and anomalies.
"""

import os
import sys
import numpy as np
import pandas as pd

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

def run_eda(train_path="data/train.csv", val_path="data/validation.csv"):
    print("=" * 60)
    print("1. LOADING DATASETS")
    print("=" * 60)
    train_df = pd.read_csv(train_path, header=None, names=TRAIN_VAL_COLS, dtype=str, keep_default_na=False)
    val_df = pd.read_csv(val_path, header=None, names=TRAIN_VAL_COLS, dtype=str, keep_default_na=False)

    print(f"Train records: {len(train_df):,}")
    print(f"Validation records: {len(val_df):,}")

    train_df["attack_cat"] = train_df["attack_cat"].str.strip().replace({"": "Normal"})
    val_df["attack_cat"] = val_df["attack_cat"].str.strip().replace({"": "Normal"})
    train_df["Label"] = pd.to_numeric(train_df["Label"], errors="coerce").fillna(0).astype(int)
    val_df["Label"] = pd.to_numeric(val_df["Label"], errors="coerce").fillna(0).astype(int)

    print("\n" + "=" * 60)
    print("2. TARGET DISTRIBUTION (LABEL 0 vs 1)")
    print("=" * 60)
    print("Train Label counts:")
    train_counts = train_df["Label"].value_counts()
    print(train_counts)
    print("Train attack proportion: {:.2%}".format(train_df["Label"].mean()))

    print("\nValidation Label counts:")
    val_counts = val_df["Label"].value_counts()
    print(val_counts)
    print("Validation attack proportion: {:.2%}".format(val_df["Label"].mean()))

    print("\n" + "=" * 60)
    print("3. ATTACK CATEGORY DISTRIBUTION")
    print("=" * 60)
    print("Train attack categories:")
    print(train_df["attack_cat"].value_counts())
    print("\nValidation attack categories:")
    print(val_df["attack_cat"].value_counts())

    print("\n" + "=" * 60)
    print("4. CATEGORICAL FEATURE CARDINALITY & OVERLAP")
    print("=" * 60)
    for col in CATEGORICAL_COLS:
        train_cats = set(train_df[col].unique())
        val_cats = set(val_df[col].unique())
        print(f"Column '{col}':")
        print(f"  Train unique: {len(train_cats)}, Val unique: {len(val_cats)}")
        unseen_in_val = val_cats - train_cats
        print(f"  Unseen categories in validation: {len(unseen_in_val)} -> {list(unseen_in_val)[:10]}")

if __name__ == "__main__":
    train_file = sys.argv[1] if len(sys.argv) > 1 else "data/train.csv"
    val_file = sys.argv[2] if len(sys.argv) > 2 else "data/validation.csv"
    run_eda(train_file, val_file)
