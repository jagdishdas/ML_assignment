"""
src/train.py — End-to-end training, cross-validation, and prediction script.

Run from the repository root:
    python src/train.py

Outputs
-------
  outputs/cv_results.csv                   – per-fold and averaged CV metrics
  validation_predictions.csv               – 12,000 final predictions
  december_chart_inputs_completed.csv      – 31 December route predictions

Assumptions for December predictions
--------------------------------------
  * market_index : per-date median from validation.csv inputs (Nov–Dec 2025).
  * quote_signal : per-date median from validation.csv inputs.
  * Lexington lat/lon  : (36.99152, −84.99876)  from data/train_test.csv.
  * Fort Wayne lat/lon : (41.31561, −85.36206)  from data/train_test.csv.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

# Resolve project root so script works whether run from root or src/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features import (
    FEATURE_NAMES,
    RPM_HIGH_OUTLIER,
    RPM_LOW_OUTLIER,
    build_features,
    haversine_distance,
    rpm_outlier_mask,
)

warnings.filterwarnings("ignore")

# ─── Configuration ────────────────────────────────────────────────────────────
RANDOM_STATE = 42

TRAIN_PATH    = ROOT / "data" / "train_test.csv"
VAL_PATH      = ROOT / "data" / "validation.csv"
TEMPLATE_PATH = ROOT / "data" / "validation_predictions_template.csv"
DEC_INPUT     = ROOT / "data" / "december_chart_inputs.csv"

OUT_CV         = ROOT / "outputs" / "cv_results.csv"
OUT_PREDS      = ROOT / "validation_predictions.csv"
OUT_DEC        = ROOT / "december_chart_inputs_completed.csv"

# Rolling-origin CV folds: (train_end, val_start, val_end) — month boundaries
CV_FOLDS = [
    ("2025-06-30", "2025-07-01", "2025-08-31"),   # fold 1
    ("2025-07-31", "2025-08-01", "2025-09-30"),   # fold 2
    ("2025-08-31", "2025-09-01", "2025-10-31"),   # fold 3
]

# HGBR hyper-parameters (same for both loss variants)
HGBR_PARAMS = dict(
    max_iter=500,
    learning_rate=0.05,
    max_depth=6,
    min_samples_leaf=20,
    l2_regularization=0.1,
    random_state=RANDOM_STATE,
    early_stopping=False,
)


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and parse data/train_test.csv and data/validation.csv."""
    train = pd.read_csv(TRAIN_PATH, parse_dates=["date"])
    val   = pd.read_csv(VAL_PATH,   parse_dates=["date"])
    print(f"  Train rows : {len(train):,}  ({train['date'].min().date()} – {train['date'].max().date()})")
    print(f"  Val rows   : {len(val):,}  ({val['date'].min().date()} – {val['date'].max().date()})")
    return train, val


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "",
) -> dict:
    """
    Compute MAE, RMSE, MAPE, MedAPE, WAPE on dollar-valued predictions.

    All inputs are in $ (actual posted rates, not log-transformed).
    """
    ae   = np.abs(y_pred - y_true)
    ape  = ae / np.abs(y_true)
    return {
        "label"    : label,
        "n"        : len(y_true),
        "mae"      : float(np.mean(ae)),
        "rmse"     : float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "mape"     : float(np.mean(ape) * 100),
        "med_ape"  : float(np.median(ape) * 100),
        "wape"     : float(np.sum(ae) / np.sum(np.abs(y_true)) * 100),
    }


def eval_all_and_noo(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    distance: np.ndarray,
    label: str,
) -> dict:
    """
    Evaluate metrics for (a) all rows and (b) non-outlier rows.
    Outlier = RPM < 1.2 or RPM > 4.5 in y_true.
    """
    rpm_true = y_true / distance
    noo_mask = (rpm_true >= RPM_LOW_OUTLIER) & (rpm_true <= RPM_HIGH_OUTLIER)

    m_all = compute_metrics(y_true,          y_pred,          label + "_all")
    m_noo = compute_metrics(y_true[noo_mask], y_pred[noo_mask], label + "_noo")

    return {"all": m_all, "noo": m_noo}


# ─── Naive baseline ───────────────────────────────────────────────────────────

def naive_baseline_predict(train_df: pd.DataFrame, val_df: pd.DataFrame) -> np.ndarray:
    """
    Predict rate = median(RPM) × distance, where RPM is computed from
    non-outlier training rows only.
    """
    rpm = train_df["posted_rate"] / train_df["distance"]
    noo = (rpm >= RPM_LOW_OUTLIER) & (rpm <= RPM_HIGH_OUTLIER)
    median_rpm = float(rpm[noo].median())
    return median_rpm * val_df["distance"].values


# ─── Cross-validation ─────────────────────────────────────────────────────────

def run_cv(train_full: pd.DataFrame) -> pd.DataFrame:
    """
    Time-based rolling-origin CV with 3 expanding-window folds.
    Returns a DataFrame of per-fold metrics (+ averages).
    """
    print("\n── Rolling-Origin Cross-Validation ──────────────────────────────")
    rows = []

    for fold_idx, (train_end, val_start, val_end) in enumerate(CV_FOLDS, 1):
        fold_train = train_full[train_full["date"] <= train_end].copy()
        fold_val   = train_full[
            (train_full["date"] >= val_start) &
            (train_full["date"] <= val_end)
        ].copy()

        print(f"\n  Fold {fold_idx}: train ≤ {train_end}  |  val {val_start}–{val_end}")
        print(f"           train n={len(fold_train):,}  val n={len(fold_val):,}")

        # ── Feature engineering (fit on train, apply to val) ──────────
        # Exclude RPM outliers from training only
        outlier_mask = rpm_outlier_mask(fold_train["posted_rate"], fold_train["distance"])
        train_clean = fold_train[~outlier_mask].copy()

        X_tr, _, arts = build_features(train_clean, fit=True)
        X_va, _, _    = build_features(fold_val,    fit=False, artifacts=arts)

        # ── Targets: log(RPM) ─────────────────────────────────────────
        rpm_tr = train_clean["posted_rate"] / train_clean["distance"]
        y_tr   = np.log(rpm_tr.values)

        dist_va   = fold_val["distance"].values
        y_va_true = fold_val["posted_rate"].values

        # ── Model 0: Naive baseline ────────────────────────────────────
        preds_naive = naive_baseline_predict(fold_train, fold_val)
        m0 = eval_all_and_noo(y_va_true, preds_naive, dist_va, "naive")

        # ── Model 1: HGBR squared_error ───────────────────────────────
        m1_hgbr = HistGradientBoostingRegressor(loss="squared_error", **HGBR_PARAMS)
        m1_hgbr.fit(X_tr, y_tr)
        preds_sq = np.exp(m1_hgbr.predict(X_va)) * dist_va
        m1 = eval_all_and_noo(y_va_true, preds_sq, dist_va, "hgbr_sq")

        # ── Model 2: HGBR absolute_error ──────────────────────────────
        m2_hgbr = HistGradientBoostingRegressor(loss="absolute_error", **HGBR_PARAMS)
        m2_hgbr.fit(X_tr, y_tr)
        preds_ae = np.exp(m2_hgbr.predict(X_va)) * dist_va
        m2 = eval_all_and_noo(y_va_true, preds_ae, dist_va, "hgbr_ae")

        print(
            f"           Naive  MAE={m0['all']['mae']:,.0f}  WAPE={m0['all']['wape']:.1f}%"
            f"  (non-outlier WAPE={m0['noo']['wape']:.1f}%)"
        )
        print(
            f"           HGBR/sq MAE={m1['all']['mae']:,.0f}  WAPE={m1['all']['wape']:.1f}%"
            f"  (non-outlier WAPE={m1['noo']['wape']:.1f}%)"
        )
        print(
            f"           HGBR/ae MAE={m2['all']['mae']:,.0f}  WAPE={m2['all']['wape']:.1f}%"
            f"  (non-outlier WAPE={m2['noo']['wape']:.1f}%)"
        )

        for model_tag, metrics in [("naive", m0), ("hgbr_sq", m1), ("hgbr_ae", m2)]:
            for subset, m in metrics.items():
                rows.append({
                    "fold"      : fold_idx,
                    "model"     : model_tag,
                    "subset"    : subset,
                    "train_end" : train_end,
                    "val_start" : val_start,
                    "val_end"   : val_end,
                    "n"         : m["n"],
                    "mae"       : round(m["mae"], 2),
                    "rmse"      : round(m["rmse"], 2),
                    "mape"      : round(m["mape"], 4),
                    "med_ape"   : round(m["med_ape"], 4),
                    "wape"      : round(m["wape"], 4),
                })

    cv_df = pd.DataFrame(rows)

    # ── Averages across folds ─────────────────────────────────────────
    avg_rows = (
        cv_df.groupby(["model", "subset"])[["mae", "rmse", "mape", "med_ape", "wape"]]
        .mean()
        .round(4)
        .reset_index()
        .assign(fold="avg", n=None, train_end="", val_start="", val_end="")
    )
    cv_df = pd.concat([cv_df, avg_rows], ignore_index=True)

    return cv_df


# ─── Final model fit ──────────────────────────────────────────────────────────

def train_final(
    train_full: pd.DataFrame,
    loss: str = "squared_error",
) -> tuple[HistGradientBoostingRegressor, dict]:
    """
    Fit the final model on all 48,000 labeled rows.

    Outliers are excluded from training (but kept in evaluation).
    Returns (fitted_model, fitted_artifacts).
    """
    print(f"\n── Final Model Training (loss={loss}, n_train=48,000) ───────────")

    outlier_mask = rpm_outlier_mask(train_full["posted_rate"], train_full["distance"])
    train_clean  = train_full[~outlier_mask].copy()
    n_excluded   = outlier_mask.sum()
    print(f"  Outliers excluded from training: {n_excluded}")
    print(f"  Clean training rows            : {len(train_clean):,}")

    X_tr, _, arts = build_features(train_clean, fit=True)

    rpm_tr = train_clean["posted_rate"] / train_clean["distance"]
    y_tr   = np.log(rpm_tr.values)

    model = HistGradientBoostingRegressor(loss=loss, **HGBR_PARAMS)
    model.fit(X_tr, y_tr)

    print(f"  Training complete.  n_iterations={model.n_iter_}")
    return model, arts


# ─── Predict validation set ───────────────────────────────────────────────────

def predict_validation(
    model: HistGradientBoostingRegressor,
    arts: dict,
    val_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply the final model to validation.csv and produce the submission CSV.

    Returns DataFrame with columns [load_id, predicted_rate].
    """
    print("\n── Predicting Validation Set ────────────────────────────────────")

    X_va, _, _ = build_features(val_df, fit=False, artifacts=arts)
    log_rpm_pred = model.predict(X_va)
    predicted_rate = np.exp(log_rpm_pred) * val_df["distance"].values

    # Clip to strictly positive (safety net; should not trigger)
    predicted_rate = np.maximum(predicted_rate, 1.0)

    template = pd.read_csv(TEMPLATE_PATH)
    result   = template[["load_id"]].copy()
    result["predicted_rate"] = predicted_rate

    # Sanity checks
    assert len(result) == 12_000, "Wrong row count"
    assert (result["predicted_rate"] > 0).all(), "Non-positive predictions"
    assert not result["load_id"].duplicated().any(), "Duplicate IDs"

    print(f"  predicted_rate  min={predicted_rate.min():.2f}  "
          f"max={predicted_rate.max():.2f}  mean={predicted_rate.mean():.2f}")
    return result


# ─── December predictions ─────────────────────────────────────────────────────

def predict_december(
    model: HistGradientBoostingRegressor,
    arts: dict,
    val_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate predictions for every day in December 2025 on the fixed lane:
      Lexington → Fort Wayne | 360 mi | Dry Van | 32,000 lb

    Assumptions (stated in assessment report):
      * market_index : per-date median from validation.csv inputs
      * quote_signal : per-date median from validation.csv inputs
      * Coordinates  : from data/train_test.csv lookup
    """
    print("\n── December Predictions ─────────────────────────────────────────")

    # December date range
    dec_dates = pd.date_range("2025-12-01", "2025-12-31", freq="D")

    # Per-date medians from validation.csv for December
    val_dec = val_df[val_df["date"].dt.month == 12].copy()
    mi_map  = val_dec.groupby("date")["market_index"].median().to_dict()
    qs_map  = val_dec.groupby("date")["quote_signal"].median().to_dict()

    # Fallback: median of all val market_index / quote_signal
    mi_global = float(val_df["market_index"].dropna().median())
    qs_global = float(val_df["quote_signal"].median())

    # Build synthetic rows for December 1-31
    dec_rows = []
    for d in dec_dates:
        dec_rows.append({
            "load_id"      : f"DEC-{d.strftime('%Y-%m-%d')}",
            "pickup"       : "Lexington",
            "delivery"     : "Fort Wayne",
            "pickup_lat"   : 36.99152,
            "pickup_lon"   : -84.99876,
            "delivery_lat" : 41.31561,
            "delivery_lon" : -85.36206,
            "distance"     : 360.0,
            "equipment"    : "Dry Van",
            "weight"       : 32_000.0,
            "date"         : d,
            "market_index" : mi_map.get(d, mi_global),
            "quote_signal" : qs_map.get(d, qs_global),
        })
    dec_df = pd.DataFrame(dec_rows)

    # Predict (use fitted artifacts from final model)
    X_dec, _, _ = build_features(dec_df, fit=False, artifacts=arts)
    log_rpm_pred = model.predict(X_dec)
    predicted_rate = np.exp(log_rpm_pred) * 360.0

    print(f"  Dec predicted_rate  min={predicted_rate.min():.2f}  "
          f"max={predicted_rate.max():.2f}  mean={predicted_rate.mean():.2f}")

    # Assemble final CSV (keep original 7 columns + predicted_rate filled in)
    dec_input = pd.read_csv(DEC_INPUT)
    dec_input["date"] = pd.to_datetime(dec_input["date"])
    dec_input = dec_input.sort_values("date").reset_index(drop=True)
    dec_input["predicted_rate"] = predicted_rate
    # Restore column order expected by score.py
    dec_input = dec_input[["pickup", "delivery", "distance", "equipment",
                             "weight", "date", "predicted_rate"]]
    dec_input["date"] = dec_input["date"].dt.strftime("%Y-%m-%d")
    dec_input["distance"] = 360.0
    dec_input["weight"] = 32000.0

    return dec_input


# ─── Determine best model from CV ────────────────────────────────────────────

def pick_best_loss(cv_df: pd.DataFrame) -> str:
    """
    Compare averaged WAPE (non-outlier subset) for hgbr_sq vs hgbr_ae.
    Returns the loss name of the better model.
    """
    avg = cv_df[cv_df["fold"] == "avg"]
    noo = avg[avg["subset"] == "noo"][["model", "wape"]].set_index("model")["wape"]
    sq_wape = noo.get("hgbr_sq", np.inf)
    ae_wape = noo.get("hgbr_ae", np.inf)
    print(f"\n  CV avg WAPE (non-outlier): hgbr_sq={sq_wape:.4f}%  hgbr_ae={ae_wape:.4f}%")
    if ae_wape < sq_wape:
        print("  → Selecting absolute_error loss (lower WAPE)")
        return "absolute_error"
    print("  → Selecting squared_error loss (lower or equal WAPE)")
    return "squared_error"


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("  Freight Rate Prediction — Training & Prediction Pipeline")
    print("=" * 65)

    # 1. Load data
    print("\n── Loading Data ──────────────────────────────────────────────────")
    train_df, val_df = load_data()

    # 2. Cross-validation
    cv_df = run_cv(train_df)

    # Save CV results
    (ROOT / "outputs").mkdir(parents=True, exist_ok=True)
    cv_df.to_csv(OUT_CV, index=False)
    print(f"\n  CV results saved → {OUT_CV}")

    # 3. Pick best loss
    best_loss = pick_best_loss(cv_df)

    # 4. Train final model on all 48k rows
    final_model, arts = train_final(train_df, loss=best_loss)

    # 5. Predict validation set
    val_preds = predict_validation(final_model, arts, val_df)
    val_preds.to_csv(OUT_PREDS, index=False)
    print(f"  Validation predictions saved → {OUT_PREDS}")

    # 6. Predict December chart
    dec_preds = predict_december(final_model, arts, val_df)
    dec_preds.to_csv(OUT_DEC, index=False)
    print(f"  December predictions saved  → {OUT_DEC}")

    print("\n" + "=" * 65)
    print("  Pipeline complete. Run score.py to validate and chart.")
    print("=" * 65)


if __name__ == "__main__":
    main()
