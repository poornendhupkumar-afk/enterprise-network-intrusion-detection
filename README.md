# Enterprise Network Intrusion Detection System (IDS)
### ML Hackathon Solution & Pipeline Reference Guide

An enterprise-grade, high-throughput Machine Learning Intrusion Detection System built for SOC environments processing hundreds of millions of network flow records daily.

---

## 🎯 Executive Overview & Challenges Solved

| Challenge | Real-World Network Context | Engineered Solution |
| :--- | :--- | :--- |
| **Asymmetric Operational Risk** | Missing an intrusion ($\text{FN}$) is catastrophic compared to a false alarm ($\text{FP}$) | Rigorous empirical threshold optimization minimizing $\text{Cost} = 40 \times \text{FN} + \text{FP}$ ($T^* = 0.0086$), reducing operational cost by **91.1%** with **99.99% recall** (only 4 missed intrusions out of 44,040). |
| **Evolving Adversaries & Zero-Days** | Novel attack families and exploits absent from training traffic | Invariant ratio features, structural protocol anomaly indicators, and zero-window flags that catch anomalies regardless of signature memorization. |
| **Dynamic Operating Environments** | Link bandwidth variations, latency spikes, and network congestion | Dimensionless features (byte ratios, packet ratios, loss proportions, TCP RTT ratios) invariant to raw throughput scaling shifts. |
| **Anti-Shortcut Forensics** | Removed 9 leaky testbed features (`srcip`, `sport`, `dstip`, `dsport`, `Stime`, `Ltime`, `sttl`, `dttl`, `ct_state_ttl`) | Learns genuine flow behavior, connection density, and protocol dynamics rather than IP/port memorization. |

---

## 📊 Performance Benchmarks (Full Validation Set: 381,007 records | Cost = 40 * FN + FP)

| Model Pipeline | Decision Threshold ($T$) | Recall (Catch Rate) | Precision | F1-Score | PR-AUC | False Negatives ($\text{FN}$) | False Positives ($\text{FP}$) | Operational Cost ($40 \times \text{FN} + \text{FP}$) | Cost Reduction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard Baseline** | $0.5000$ | $97.76\%$ | $98.49\%$ | $0.9813$ | $0.9990$ | $985$ | $659$ | $40,059$ | Benchmark |
| **Cost-Tuned Baseline** | $0.0276$ | $99.98\%$ | $93.08\%$ | $0.9640$ | $0.9990$ | $8$ | $3,276$ | $3,596$ | $-91.0\%$ |
| **Re-Tuned Ensemble (Cost 40)** | **$0.0086$** | **$99.99\%$** | **$92.85\%$** | **$0.9629$** | **$0.9990$** | **4** | **3,393** | **3,553** | **-91.1%** |

### Attack Category Detection Breakdown (Ensemble @ $T^* = 0.0086$)
- **Analysis:** **$100.00\%$** ($59 / 59$)
- **Exploits:** **$99.91\%$** ($3,404 / 3,407$)
- **Fuzzers:** **$99.95\%$** ($2,216 / 2,217$)
- **Generic:** **$100.00\%$** ($38,332 / 38,332$)
- **Worms:** **$100.00\%$** ($25 / 25$)
- **Normal Specificity:** **$98.99\%$** ($333,574 / 336,967$ legitimate traffic passed without alert)

---

## 🛠️ Quickstart Commands

### 1. Model Inference (Official Competition Format)
To run predictions on any unlabelled evaluation dataset (`challenge.csv` or `final_test.csv`):
```bash
python predict.py --input path/to/challenge.csv --output submission_challenge.csv --model model.joblib
```

### 2. Validate and Evaluate Locally
To validate submission format and evaluate on labeled validation data:
```bash
python evaluate.py --submission submissions/submission_val_advanced_full.csv --ground-truth data/validation.csv
```

### 3. Retrain Full Advanced Ensemble
```bash
python train_advanced.py --train data/train.csv --val data/validation.csv --save-model models/advanced_model_full.joblib
```

---

## 📁 Repository Structure

```
ids_hackathon/
├── data/                               # Dataset directory (train.csv, validation.csv)
├── models/                             # Saved model artifacts (.joblib)
│   ├── advanced_model_full.joblib      # Production ensemble artifact
│   └── baseline_full.joblib            # Baseline benchmark artifact
├── submissions/                        # Output prediction CSV files
│   └── submission_val_advanced_full.csv
├── model.joblib                        # Symlinked / top-level production artifact
├── features.py                         # Reusable feature engineering module
├── predict.py                          # Official competition inference script
├── evaluate.py                         # Local evaluation and format validator
├── train_advanced.py                   # High-performance multi-GBDT training pipeline
├── baseline.py                         # Starter baseline pipeline
└── eda.py                              # Exploratory data analysis utility
```
