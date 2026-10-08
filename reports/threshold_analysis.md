# Validation threshold analysis

## Method

Loaded the existing `model.joblib` ensemble and ran inference only; no fitting or model-artifact changes were made. Feature processing uses the model's stored categorical encodings and `features.extract_features`. Probabilities are generated exactly as in `predict.py`: `0.60 * LightGBM positive-class probability + 0.40 * XGBoost positive-class probability`. Validation data was read in chunks from `data/validation.csv` and was not modified.

The CSV evaluates 695 classification thresholds, with a fine 0.0001 grid from 0.0001 to 0.05 and 0.005 grid from 0.055 to 1.0, plus the recorded thresholds and the exact best threshold. To guarantee the global minimum, cost was additionally assessed at every distinct model score boundary (319,379 values); threshold decisions only change at these boundaries.

## Minimum-cost threshold

| Metric | Value |
|---|---:|
| Threshold | 0.0175 |
| TP | 44,036 |
| FP | 3,316 |
| TN | 333,651 |
| FN | 4 |
| Precision | 0.929971 |
| Recall | 0.999909 |
| F1 | 0.963673 |
| Cost (`40 * FN + FP`) | 3,476 |

The minimum is tied across a plateau: the tested thresholds from 0.0175 through 0.0178 (and the exact score-boundary value 0.01788612358) produce the same predictions and cost. The reported threshold is the first grid value attaining that minimum.

Confusion matrix (actual rows `[Normal, Attack]`, predicted columns `[Normal, Attack]`):

```text
[[333,651, 3,316],
 [4, 44,036]]
```

## Comparison to the current result

The supplied current operating result is TP=44,036, FP=3,393, TN=333,574, FN=4, cost=3,553. Against that result:

- Cost improvement: **77** (2.17%).
- Additional attacks missed: **0**.
- False alarms eliminated: **77**.

Threshold provenance check: the saved artifact's `best_threshold` is **0.0176158079**, not 0.0086. At 0.0086 this validation run produced TP=44,036, FP=3,396, TN=333,571, FN=4, cost=3,556. At the saved artifact threshold it produced TP=44,036, FP=3,316, TN=333,651, FN=4, cost=3,476. Thus the supplied baseline confusion matrix does not correspond to either measured operating point for the currently saved artifact; comparison deltas above are calculated against the supplied cost/counts as requested.

Against the model's measured 0.0086 result, the minimum reduces cost by **80**, misses **0** additional attacks, and eliminates **80** false alarms. The difference between the supplied baseline and the measured 0.0086 result (3 false alarms) is reported rather than silently reconciled.

All evaluated threshold metrics are in [`threshold_analysis.csv`](./threshold_analysis.csv).
