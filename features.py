#!/usr/bin/env python3
"""
features.py
===========
Unified feature engineering module for Network Intrusion Detection.
Ensures 100% consistent feature transformations across training and inference.
"""

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
CATEGORICAL_COLS = ["proto", "state", "service"]
NUMERIC_COLS = [c for c in FEATURE_NAMES if c not in CATEGORICAL_COLS]


def extract_features(df, cat_levels=None, is_train=False):
    """
    Extracts all raw and domain-engineered features from input DataFrame.
    """
    X = pd.DataFrame(index=df.index)

    # 1. Categorical encoding
    if is_train or cat_levels is None:
        cat_levels = {}
        for c in CATEGORICAL_COLS:
            unique_vals = sorted(df[c].astype(str).unique())
            cat_levels[c] = {val: idx for idx, val in enumerate(unique_vals)}

    for c in CATEGORICAL_COLS:
        mapping = cat_levels[c]
        oov_idx = len(mapping)
        X[c] = df[c].astype(str).map(mapping).fillna(oov_idx).astype(int)

    # 2. Raw numeric features
    for c in NUMERIC_COLS:
        X[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(np.float32)

    # 3. Domain Engineered Features
    sbytes = X["sbytes"].values
    dbytes = X["dbytes"].values
    spkts = X["Spkts"].values
    dpkts = X["Dpkts"].values
    smeansz = X["smeansz"].values
    dmeansz = X["dmeansz"].values
    sloss = X["sloss"].values
    dloss = X["dloss"].values
    sload = X["Sload"].values
    dload = X["Dload"].values
    sjit = X["Sjit"].values
    djit = X["Djit"].values
    sintpkt = X["Sintpkt"].values
    dintpkt = X["Dintpkt"].values
    dur = X["dur"].values
    tcprtt = X["tcprtt"].values
    synack = X["synack"].values
    ackdat = X["ackdat"].values
    swin = X["swin"].values
    dwin = X["dwin"].values
    stcpb = X["stcpb"].values
    dtcpb = X["dtcpb"].values

    ct_dst_ltm = X["ct_dst_ltm"].values
    ct_src_ltm = X["ct_src_ltm"].values
    ct_srv_dst = X["ct_srv_dst"].values
    ct_srv_src = X["ct_srv_src"].values
    ct_dst_sport_ltm = X["ct_dst_sport_ltm"].values
    ct_src_dport_ltm = X["ct_src_dport_ltm"].values
    ct_dst_src_ltm = X["ct_dst_src_ltm"].values

    # Traffic ratios (Dimensionless & Robust to link bandwidth shifts)
    X["tot_bytes"] = (sbytes + dbytes).astype(np.float32)
    X["tot_pkts"] = (spkts + dpkts).astype(np.float32)
    X["byte_ratio"] = ((sbytes + 1.0) / (dbytes + 1.0)).astype(np.float32)
    X["pkt_ratio"] = ((spkts + 1.0) / (dpkts + 1.0)).astype(np.float32)
    X["src_bytes_per_pkt"] = (sbytes / (spkts + 1.0)).astype(np.float32)
    X["dst_bytes_per_pkt"] = (dbytes / (dpkts + 1.0)).astype(np.float32)
    X["pkt_size_asymmetry"] = (np.abs(smeansz - dmeansz) / (smeansz + dmeansz + 1.0)).astype(np.float32)

    # Loss dynamics
    X["tot_loss"] = (sloss + dloss).astype(np.float32)
    X["src_loss_ratio"] = (sloss / (spkts + 1.0)).astype(np.float32)
    X["dst_loss_ratio"] = (dloss / (dpkts + 1.0)).astype(np.float32)
    X["tot_loss_ratio"] = ((sloss + dloss) / (spkts + dpkts + 1.0)).astype(np.float32)

    # Load and timing
    X["load_ratio"] = ((sload + 1.0) / (dload + 1.0)).astype(np.float32)
    X["jitter_ratio"] = ((sjit + 1e-4) / (djit + 1e-4)).astype(np.float32)
    X["intpkt_ratio"] = ((sintpkt + 1e-4) / (dintpkt + 1e-4)).astype(np.float32)
    X["pkt_rate"] = ((spkts + dpkts) / (dur + 1e-5)).astype(np.float32)
    X["byte_rate"] = ((sbytes + dbytes) / (dur + 1e-5)).astype(np.float32)

    # TCP connection handshake ratios
    X["tcprtt_ratio_synack"] = (synack / (tcprtt + 1e-6)).astype(np.float32)
    X["tcprtt_ratio_ackdat"] = (ackdat / (tcprtt + 1e-6)).astype(np.float32)

    # TCP flags / Zero-window indicators
    X["is_swin_0"] = (swin == 0).astype(np.float32)
    X["is_dwin_0"] = (dwin == 0).astype(np.float32)
    X["is_swin_255"] = (swin == 255).astype(np.float32)
    X["is_dwin_255"] = (dwin == 255).astype(np.float32)
    X["is_stcpb_0"] = (stcpb == 0).astype(np.float32)
    X["is_dtcpb_0"] = (dtcpb == 0).astype(np.float32)

    # Scan & Reconnaissance Indicators
    X["port_scan_ratio"] = ((ct_dst_sport_ltm + 1.0) / (ct_dst_ltm + 1.0)).astype(np.float32)
    X["host_scan_ratio"] = ((ct_src_dport_ltm + 1.0) / (ct_src_ltm + 1.0)).astype(np.float32)
    X["srv_src_ratio"] = ((ct_srv_src + 1.0) / (ct_src_ltm + 1.0)).astype(np.float32)
    X["srv_dst_ratio"] = ((ct_srv_dst + 1.0) / (ct_dst_ltm + 1.0)).astype(np.float32)
    X["src_dst_conn_ratio"] = ((ct_dst_src_ltm + 1.0) / (ct_dst_ltm + 1.0)).astype(np.float32)

    # Log transforms for skewed volume features
    X["log_dur"] = np.log1p(np.maximum(0.0, dur)).astype(np.float32)
    X["log_sbytes"] = np.log1p(np.maximum(0.0, sbytes)).astype(np.float32)
    X["log_dbytes"] = np.log1p(np.maximum(0.0, dbytes)).astype(np.float32)
    X["log_Sload"] = np.log1p(np.maximum(0.0, sload)).astype(np.float32)
    X["log_Dload"] = np.log1p(np.maximum(0.0, dload)).astype(np.float32)

    # 4. Fine-grained features for false-positive reduction
    service_str = df["service"].astype(str).values
    proto_str = df["proto"].astype(str).values
    state_str = df["state"].astype(str).values

    X["is_service_dash"] = (service_str == "-").astype(np.float32)
    X["is_service_http"] = (service_str == "http").astype(np.float32)
    X["is_service_dns"] = (service_str == "dns").astype(np.float32)
    X["is_state_FIN"] = (state_str == "FIN").astype(np.float32)
    X["is_state_INT"] = (state_str == "INT").astype(np.float32)
    X["is_proto_tcp"] = (proto_str == "tcp").astype(np.float32)
    X["is_proto_udp"] = (proto_str == "udp").astype(np.float32)

    # Directional & probe features
    X["is_unidirectional"] = (dpkts == 0).astype(np.float32)
    X["is_single_pkt_probe"] = ((spkts == 1) & (dpkts == 0)).astype(np.float32)
    X["byte_diff"] = (sbytes - dbytes).astype(np.float32)
    X["pkt_diff"] = (spkts - dpkts).astype(np.float32)
    X["mean_sz_ratio"] = ((smeansz + 1.0) / (dmeansz + 1.0)).astype(np.float32)

    # Multi-dimensional density ratios
    X["srv_to_dst_ratio"] = ((ct_srv_dst + 1.0) / (ct_dst_ltm + 1.0)).astype(np.float32)
    X["srv_src_to_dst_ratio"] = ((ct_srv_src + 1.0) / (ct_srv_dst + 1.0)).astype(np.float32)
    X["scan_entropy_indicator"] = ((ct_dst_sport_ltm + 1.0) / (ct_src_dport_ltm + 1.0)).astype(np.float32)
    X["conn_concentration"] = ((ct_dst_src_ltm + 1.0) / (ct_srv_src + 1.0)).astype(np.float32)

    # HTTP & FTP specific indicators
    trans_depth = X["trans_depth"].values
    res_bdy_len = X["res_bdy_len"].values
    ct_flw_http_mthd = X["ct_flw_http_mthd"].values
    is_ftp_login = X["is_ftp_login"].values
    ct_ftp_cmd = X["ct_ftp_cmd"].values

    X["has_res_body"] = (res_bdy_len > 0).astype(np.float32)
    X["res_body_to_dbytes"] = (res_bdy_len / (dbytes + 1.0)).astype(np.float32)
    X["is_http_active"] = ((ct_flw_http_mthd > 0) | (trans_depth > 0) | (res_bdy_len > 0)).astype(np.float32)
    X["is_ftp_active"] = ((is_ftp_login > 0) | (ct_ftp_cmd > 0)).astype(np.float32)

    return X, cat_levels
