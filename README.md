# Freight Rate Prediction — ML Assessment

## Overview

Predicts the `posted_rate` (freight load price in USD) for 12,000 unseen loads
using a `HistGradientBoostingRegressor` trained on 48,000 labeled loads
(`data/train_test.csv`). The model targets `log(rate-per-mile)` and recovers the
dollar prediction via `exp(output) × distance`.

---

## Setup

### Requirements

- Python 3.9+
- Packages: `pandas`, `numpy`, `matplotlib`, `scikit-learn`

### Install

```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

All commands must be run from the **repository root** (the `ML_assignment/` directory).

### 1. Train and generate predictions

```bash
python src/train.py
```

This produces:
- `outputs/cv_results.csv` — 3-fold rolling-origin cross-validation metrics
- `validation_predictions.csv` — 12,000 final rate predictions
- `december_chart_inputs_completed.csv` — 31 December daily predictions

Optional — re-run the feature ablation used in the report (writes `outputs/ablation_results.csv`):

```bash
python src/ablation.py
```

### 2. Validate outputs and generate the December chart

```bash
python score.py \
  --predictions validation_predictions.csv \
  --december-predictions december_chart_inputs_completed.csv \
  --output-dir scorer_results
```

This produces `scorer_results/candidate_december.png`.

---

## Repository Structure

```
ML_assignment/
├── src/
│   ├── features.py            Feature engineering (cleaning, encoding, imputation)
│   ├── train.py               CV, model training, prediction pipeline
│   └── ablation.py            Feature / recency ablation over the same CV folds
├── outputs/
│   ├── cv_results.csv         Cross-validation results table
│   └── ablation_results.csv   Ablation results table
├── scorer_results/
│   └── candidate_december.png December rate chart
├── data/
│   ├── train_test.csv                     48,000 labeled training loads (Jan–Oct 2025)
│   ├── validation.csv                     12,000 unlabeled loads (Nov–Dec 2025)
│   ├── validation_predictions_template.csv  Submission template
│   └── december_chart_inputs.csv          31-day fixed-lane December inputs
├── validation_predictions.csv  ← SUBMISSION FILE
├── december_chart_inputs_completed.csv  ← DECEMBER FILE
├── score.py                   Assessment validator (do not modify)
├── requirements.txt
├── README.md
├── make_report_docx.py        Rebuilds report.docx from the result files
└── report.docx                Write-up (validation approach + December chart)
```

---

## Modeling Approach (brief)

| Decision | Rationale |
|----------|-----------|
| Target: `log(RPM)` | RPM is strongly right-skewed (skew 8.72 → −0.04 after log); prediction = `exp × dist`, always positive |
| No `month_of_year` feature | A month code means nothing for unseen months (Nov–Dec) |
| `day_index` (days since 2025-01-01) + recency weights `exp(-age/90 d)` | Later dates map to the latest training period; the model follows the current market level. Removed a ~2.5% under-prediction seen in every CV fold |
| Day-level market features (`day_mi_median`, `day_qs_median`, `quote_rel`) | `market_index` is a daily market level; these use inputs only (no target) |
| `day_of_week`, `day_of_month` | Capture within-week and within-month patterns |
| RPM outliers (< 1.2 or > 4.5 $/mi) excluded from training | Kept in the "all rows" evaluation |
| Market index NaN → date-median from inputs | No target leakage |
| Seed = 42 everywhere | Deterministic, reproducible outputs |

Rolling-origin CV (3 expanding folds), non-outlier WAPE: naive baseline 8.76% →
original model 3.34% → final model 2.66% (see `outputs/ablation_results.csv`).
The recency constant and features were chosen on these same folds, so treat 2.66%
as somewhat optimistic.

See `report.docx` for full details.

---

## Assumptions for December Predictions

The `data/december_chart_inputs.csv` file does not contain `market_index` or
`quote_signal`. The following assumptions are applied:

1. **`market_index`**: per-date median from `validation.csv` inputs for December 2025.
2. **`quote_signal`**: per-date median from `validation.csv` inputs for December 2025.
3. **Lexington coordinates**: `(36.99152, −84.99876)` — looked up from `data/train_test.csv`.
4. **Fort Wayne coordinates**: `(41.31561, −85.36206)` — looked up from `data/train_test.csv`.
