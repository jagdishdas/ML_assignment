# Freight Rate Prediction — ML Assessment

## Overview

Predicts the `posted_rate` (freight load price in USD) for 12,000 unseen loads
using a `HistGradientBoostingRegressor` trained on 48,000 labeled loads
(`train-test.csv`). The model targets `log(rate-per-mile)` and recovers the
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
│   └── train.py               CV, model training, prediction pipeline
├── outputs/
│   └── cv_results.csv         Cross-validation results table
├── scorer_results/
│   └── candidate_december.png December rate chart
├── data/                      (empty — data files live in repo root)
├── train-test.csv             48,000 labeled training loads (Jan–Oct 2025)
├── validation.csv             12,000 unlabeled loads (Nov–Dec 2025)
├── validation-predictions-template.csv  Submission template
├── december-chart-inputs.csv  31-day fixed-lane December inputs
├── validation_predictions.csv  ← SUBMISSION FILE
├── december_chart_inputs_completed.csv  ← DECEMBER FILE
├── score.py                   Assessment validator (do not modify)
├── requirements.txt
├── README.md
└── report.pdf                 2-3 page write-up
```

---

## Modeling Approach (brief)

| Decision | Rationale |
|----------|-----------|
| Target: `log(RPM)` | Stabilises heteroscedastic variance; recovers $ via `exp × dist` |
| No `month_of_year` feature | HGBR trees cannot extrapolate to unseen months (Nov–Dec) |
| `day_of_week`, `day_of_month` | Capture within-week and within-month cyclical patterns |
| RPM outliers excluded from training | Spec requirement; kept in "all rows" evaluation |
| Market index NaN → date-median from inputs | No target leakage; matches spec |
| Seed = 42 everywhere | Deterministic, reproducible outputs |

See `report.pdf` for full details.

---

## Assumptions for December Predictions

The `december-chart-inputs.csv` file does not contain `market_index` or
`quote_signal`. The following assumptions are applied:

1. **`market_index`**: per-date median from `validation.csv` inputs for December 2025.
2. **`quote_signal`**: per-date median from `validation.csv` inputs for December 2025.
3. **Lexington coordinates**: `(36.99152, −84.99876)` — looked up from `train-test.csv`.
4. **Fort Wayne coordinates**: `(41.31561, −85.36206)` — looked up from `train-test.csv`.
