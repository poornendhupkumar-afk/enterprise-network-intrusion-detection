# Hour 4 Challenge Prediction Summary

## Dataset verification

- Input: `data/challenge.csv`
- Rows: **254,005**
- Fields per row: **38**
- Structure: headerless, with exactly 38 fields on every row
- Labels: **unlabeled**; no `attack_cat` or `Label` columns were supplied
- Compatibility: the 38 raw fields match the saved model's 38 training feature names in `features.py`

## Model and inference pipeline

- Model: saved LightGBM/XGBoost ensemble in `model.joblib` (same saved model package as `models/advanced_model_full.joblib`)
- Ensemble weights: **60% LightGBM + 40% XGBoost**
- Preprocessing and feature engineering: `features.extract_features`, using only the categorical encodings saved with the trained model. It encodes the three categorical fields (`proto`, `state`, `service`), converts numeric fields using the existing pipeline, and derives the model's 92 expected features. No preprocessing or feature transformations were fitted on challenge data.
- Threshold: **0.0175**, the minimum-cost validation threshold recorded in `reports/threshold_analysis.md`. It is within the validation minimum-cost plateau and produces the same predictions as the model artifact's stored threshold.
- Training confirmation: challenge data was used only for inference; it was **not used to train or tune any model**. `train.csv`, `validation.csv`, `challenge.csv`, and the saved model artifacts were not modified.

## Predictions

- Predicted normal (`0`): **193,360** (**76.12%**)
- Predicted attack (`1`): **60,645** (**23.88%**)
- Original row order: preserved; submission `row_id` is sequential from 1 through 254,005.

No accuracy, precision, recall, F1, confusion matrix, or cost is reported because this challenge dataset has no ground-truth labels.

## Submission

- File: `submissions/submission_challenge.csv`
- Format: `row_id,prediction,probability`
- Validation: **passed** using `evaluate.py --validate-only --expected-rows 254005`; all 254,005 rows were read, predictions were confirmed binary, and row order and probability values were checked.
