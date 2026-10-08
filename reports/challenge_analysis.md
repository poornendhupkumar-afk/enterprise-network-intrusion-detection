# Challenge Prediction and Distribution Analysis

## Challenge prediction summary

The existing `submissions/submission_challenge.csv` contains **254,005** rows with sequential row IDs.

| Predicted class | Count | Percentage |
|---|---:|---:|
| Normal (0) | 193,360 | 76.12% |
| Attack (1) | 60,645 | 23.88% |

- Threshold used: **0.0175**, the minimum-cost validation threshold.
- Model used: saved **LightGBM/XGBoost ensemble** in `model.joblib`, blended with weights 0.60/0.40.
- Validation prediction reference: `submissions/submission_val_cost40_opt.csv`, 381,007 rows, produced for the tuned validation operating point.

| Prediction | Validation | Challenge | Difference (challenge − validation) |
|---|---:|---:|---:|
| Normal count | 333,655 | 193,360 | — |
| Normal percentage | 87.57% | 76.12% | -11.45 percentage points |
| Attack count | 47,352 | 60,645 | +13,293 |
| Attack percentage | 12.43% | 23.88% | +11.45 percentage points |

The percentages use each dataset's own row count. Challenge data is unlabeled; no ground-truth-dependent performance metrics are calculated or inferred.

## Feature distribution comparison

Compared raw validation features (first 38 columns of `data/validation.csv`) with all 38 fields in `data/challenge.csv`. Both were passed through the existing `features.extract_features` function using the categorical encodings saved in `model.joblib`; no model was trained and no transformations were fit on challenge data.

Numeric features below are the ten highest-ranked non-categorical features by an ensemble importance ranking: LightGBM and XGBoost gain importances were normalized within each model and combined with their saved 0.60/0.40 weights. `SMD` is the standardized mean difference (challenge mean minus validation mean); `KS` is the two-sample Kolmogorov–Smirnov statistic. `PSI` is calculated with decile bins defined from validation values. As practical indicators, |SMD| ≥ 0.2 or PSI ≥ 0.2 is marked high; PSI 0.1–0.2 or |SMD| ≥ 0.1 is moderate. All numeric KS tests and categorical chi-square tests were significant at p < 0.001; with these sample sizes, effect sizes and PSI are more informative than p-values alone.

| Model-important numeric feature | Validation median | Challenge median | Validation 90th pct. | Challenge 90th pct. | SMD | KS | PSI | Shift |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `src_bytes_per_pkt` | 60.69 | 79.20 | 121.54 | 630.22 | +0.417 | 0.312 | 0.616 | High |
| `tot_bytes` | 6,656 | 9,501 | 54,884 | 73,438 | +0.022 | 0.157 | 0.099 | Low/modest |
| `sbytes` | 1,580 | 2,752 | 6,250 | 40,528 | +0.077 | 0.240 | 0.610 | High |
| `ackdat` | 0.000124 | 0.000111 | 0.000163 | 0.000385 | +0.158 | 0.427 | 2.852 | High |
| `tcprtt` | 0.000630 | 0.000550 | 0.000793 | 0.002623 | +0.175 | 0.423 | 2.898 | High |
| `byte_ratio` | 0.804 | 1.308 | 115.0 | 204.0 | +0.089 | 0.285 | 0.691 | High |
| `log_sbytes` | 7.366 | 7.920 | 8.740 | 10.610 | +0.469 | 0.240 | 0.610 | High |
| `Dload` | 660,482 | 363,406 | 11,210,970 | 7,379,855 | -0.228 | 0.192 | 0.266 | High |
| `synack` | 0.000497 | 0.000430 | 0.000623 | 0.002056 | +0.158 | 0.429 | 3.025 | High |
| `ct_srv_dst` | 4 | 3 | 16 | 10 | -0.354 | 0.154 | 0.177 | High |

The biggest practical shifts are in source-byte volume and per-packet features, the upper tails of handshake timing features (`ackdat`, `tcprtt`, `synack`), and service/destination load. The byte/timing and count features are skewed, so shifts in the upper quantiles and PSI can be large even where median changes are smaller.

### Categorical features

The table reports the largest category-share changes for each raw categorical feature; percentages are the share of rows in each dataset. Cramér's V summarizes the overall distribution difference (larger values indicate a stronger association with dataset membership).

| Feature | Cramér's V | Largest share changes: validation → challenge |
|---|---:|---|
| `proto` | 0.272 | `udp`: 35.14% → 22.96% (-12.19 pp); `unas`: 0.00% → 3.14%; `arp`: 0.03% → 1.84% |
| `state` | 0.103 | `CON`: 25.48% → 19.61% (-5.87 pp); `INT`: 10.24% → 13.27% (+3.03 pp); `REQ`: 0.09% → 1.02% |
| `service` | 0.297 | `dns`: 26.00% → 14.76% (-11.24 pp); `smtp`: 1.18% → 10.85% (+9.67 pp); `ssh`: 0.66% → 6.23% (+5.57 pp) |

## Interpretation

The challenge prediction distribution differs substantially from validation, with the attack prediction share higher by **11.45 percentage points**. Several high-importance features also show sizeable distribution shifts, alongside large changes in protocol and service composition. This is evidence that challenge inputs differ from validation inputs; by itself it does not establish the challenge labels or quantify model accuracy on challenge.
