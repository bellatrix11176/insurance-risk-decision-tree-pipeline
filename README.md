# Insurance Category Decision Tree (Python)

This project trains a **decision tree classifier** to predict a customer’s **InsuranceCategory** (e.g., Low Risk, Moderate Risk, Potentially High Risk, High Risk—Do Not Insure), then applies the trained model to a separate scoring dataset to generate predictions. It also saves an “evidence locker” of outputs (CSVs + charts + reports) in `output/` so results are traceable and easy to review.

---

## Overview

The script:

1. **Auto-locates** a training CSV (contains `InsuranceCategory`) and a scoring CSV (no label required) from `data/`
2. **Validates** required columns (`CustomerID`, `InsuranceCategory`) and feature consistency
3. **Preprocesses** features:
   - numeric columns: pass-through
   - categorical columns: one-hot encode (`handle_unknown="ignore"`)
4. **Runs a Python-only model selection experiment** to compare decision-tree split criteria:
   - `gini`
   - `entropy`  
   using **Stratified 5-fold cross-validation**, then selects the best criterion
5. **Trains the final model** using the selected criterion
6. **Evaluates training performance** and exports metrics + plots
7. **Scores** the scoring dataset (after filtering invalid rows) and exports predictions + counts

---

## Data

Place **two CSV files** in:

data/

### Training dataset must include
- `CustomerID` (identifier)
- `InsuranceCategory` (label/target)
- predictor columns (numeric and/or categorical)

### Scoring dataset must include
- `CustomerID`
- the **same predictor columns** as training  
  (the scoring file does not need `InsuranceCategory`)

**Important:** `CustomerID` is treated as an identifier only and is **not** used as a predictor.

---

## File auto-detection logic (no renaming required)

The script attempts, in order:

1. **Preferred filenames**
   - `insurance_risk_category_train.csv`
   - `insurance_risk_category_score.csv`

2. **Keyword-based detection**
   - training: filename contains `decision`, `tree`, `training`
   - scoring: filename contains `decision`, `tree`, `scoring`

3. **Fallback**
   - training = the CSV containing `InsuranceCategory`
   - scoring = another CSV in `data/`

---

## Range filtering (scoring data guardrail)

Before scoring, numeric predictor values are checked against training min/max ranges. Scoring rows with numeric values **outside training ranges** (or missing) are removed and saved for traceability.

Outputs:
- `training_feature_ranges_numeric.csv`
- `scoring_out_of_range_summary.csv`
- `scoring_rows_removed_out_of_range.csv` (only if rows were removed)

---

## How to run

From the project root (the folder that contains `data/`, `src/`, `output/`):

python src/insurance_category_decision_tree.py

---

## Outputs (Evidence Locker)

All outputs are written to:

output/

### Model selection
- `criterion_comparison_cv.csv`  
  Cross-validation comparison of `gini` vs `entropy`

### Performance (training)
- `performance_metrics_train.csv`
- `classification_report_train.txt`
- `confusion_matrix_train.csv`
- `confusion_matrix_train.png`

### Interpretability
- `decision_tree_top_levels.png`
- `decision_tree_full.png`
- `feature_importances.csv`
- `feature_importances_top15.png`

### Scoring / Predictions
- `insurance_category_predictions.csv`
- `prediction_counts.csv`

---

## Interpreting results

- **Selected criterion:** stored in `performance_metrics_train.csv` and `criterion_comparison_cv.csv`
- **Root feature (first split):** printed to console and visible in the tree plots
- **Low Risk count (scoring predictions):** found in:
  - console output
  - `prediction_counts.csv`

---

## Repo structure

Decision Tree/
- data/        (input CSVs: training + scoring)
- src/         (python script)
- output/      (generated artifacts)
- README.md

---

## Notes / Reproducibility

- Uses `random_state=42` for consistent results.
- One-hot encoding uses `handle_unknown="ignore"` so unseen categories in scoring data won’t crash the pipeline.
- Training accuracy is reported, while model selection is based on cross-validation for a more stable comparison.
