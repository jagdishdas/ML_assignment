"""
src/ablation.py — Where does the CV improvement come from?

Re-runs the same 3 rolling-origin folds as train.py (squared_error loss, same
hyper-parameters) while switching features / recency weighting on one step at
a time, and writes outputs/ablation_results.csv.

Run from the repository root:
    python src/ablation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features import FEATURE_NAMES, build_features, rpm_outlier_mask
from src.train import (
    CV_FOLDS, HGBR_PARAMS, RECENCY_DAYS, TRAIN_PATH,
    eval_all_and_noo, recency_weights,
)

N_BASE = 22                      # original feature set (see FEATURE_NAMES order)
N_DAY_MARKET = N_BASE + 3        # + day_mi_median, day_qs_median, quote_rel
N_ALL = len(FEATURE_NAMES)       # + day_index

# (label, number of leading FEATURE_NAMES used, recency decay constant in days or None)
CONFIGS = [
    ("A: original 22 features",                      N_BASE,       None),
    ("B: + day-level market features",               N_DAY_MARKET, None),
    ("C: + day_index",                               N_ALL,        None),
    (f"D: + recency weighting ({RECENCY_DAYS} d) [final]", N_ALL,   RECENCY_DAYS),
    ("D with recency 30 d",                          N_ALL,        30),
    ("D with recency 60 d",                          N_ALL,        60),
    ("D with recency 120 d",                         N_ALL,        120),
    ("D with recency 240 d",                         N_ALL,        240),
]


def main() -> None:
    train_full = pd.read_csv(TRAIN_PATH, parse_dates=["date"])
    rows = []
    for label, n_feat, days in CONFIGS:
        noo_wape, all_wape, bias = [], [], []
        for train_end, val_start, val_end in CV_FOLDS:
            fold_train = train_full[train_full["date"] <= train_end]
            fold_val = train_full[(train_full["date"] >= val_start) & (train_full["date"] <= val_end)]
            clean = fold_train[~rpm_outlier_mask(fold_train["posted_rate"], fold_train["distance"])]

            X_tr, _, arts = build_features(clean, fit=True)
            X_va, _, _ = build_features(fold_val, fit=False, artifacts=arts)
            y_tr = np.log((clean["posted_rate"] / clean["distance"]).values)
            w = None if days is None else recency_weights(clean["date"], days)

            model = HistGradientBoostingRegressor(loss="squared_error", **HGBR_PARAMS)
            model.fit(X_tr[:, :n_feat], y_tr, sample_weight=w)
            pred = np.exp(model.predict(X_va[:, :n_feat])) * fold_val["distance"].values
            m = eval_all_and_noo(fold_val["posted_rate"].values, pred, fold_val["distance"].values, "x")
            noo_wape.append(m["noo"]["wape"])
            all_wape.append(m["all"]["wape"])
            # mean signed % error on non-outlier rows (negative = under-prediction)
            y_true = fold_val["posted_rate"].values
            noo = ~rpm_outlier_mask(fold_val["posted_rate"], fold_val["distance"]).values
            bias.append(float(np.mean((pred[noo] - y_true[noo]) / y_true[noo]) * 100))
        rows.append({
            "config": label,
            "wape_noo_fold1": round(noo_wape[0], 4),
            "wape_noo_fold2": round(noo_wape[1], 4),
            "wape_noo_fold3": round(noo_wape[2], 4),
            "wape_noo_avg": round(float(np.mean(noo_wape)), 4),
            "wape_all_avg": round(float(np.mean(all_wape)), 4),
            "bias_noo_fold1": round(bias[0], 3),
            "bias_noo_fold2": round(bias[1], 3),
            "bias_noo_fold3": round(bias[2], 3),
            "bias_noo_avg": round(float(np.mean(bias)), 3),
        })
        print(f"  {label:<48} non-outlier WAPE avg = {np.mean(noo_wape):.4f}%")

    out = pd.DataFrame(rows)
    (ROOT / "outputs").mkdir(exist_ok=True)
    out.to_csv(ROOT / "outputs" / "ablation_results.csv", index=False)
    print("  Saved → outputs/ablation_results.csv")


if __name__ == "__main__":
    main()
