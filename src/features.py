"""
src/features.py — Feature engineering for freight-rate prediction.

All transformations are:
  - Deterministic (no randomness)
  - Leak-free (no target information used)
  - Safe for rolling-origin CV (encoders/maps fitted on training fold only)

Usage
-----
  from src.features import build_features

  # Fit on training data, get back artifacts:
  X_train, names, arts = build_features(train_df, fit=True)

  # Transform validation / test data using fitted artifacts:
  X_val, _, _ = build_features(val_df, fit=False, artifacts=arts)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ─── Constants ───────────────────────────────────────────────────────────────
EARTH_RADIUS_MILES = 3_958.8          # mean Earth radius in miles
WEIGHT_CENSOR_HIGH = 47_500.0         # right-censored upper bound
RPM_LOW_OUTLIER    = 1.2              # $/mi  – outlier threshold (lower)
RPM_HIGH_OUTLIER   = 4.5             # $/mi  – outlier threshold (upper)

# Feature columns in the exact order produced by build_features()
FEATURE_NAMES: list[str] = [
    "distance",           # reported road distance (miles)
    "hav_dist",           # haversine great-circle distance (miles)
    "dist_minus_hav",     # road − haversine (routing detour proxy)
    "weight_clean",       # weight after sign-fix and imputation
    "market_index_filled",# market-wide daily index (NaNs filled)
    "quote_signal",       # per-load price signal (~rate-per-mile)
    "quote_x_dist",       # quote_signal × distance   (rate proxy in $)
    "quote_x_hav",        # quote_signal × hav_dist   (alt rate proxy)
    "day_of_week",        # 0 = Mon … 6 = Sun
    "day_of_month",       # 1 – 31
    "is_weekend",         # 1 if Sat/Sun else 0
    "week_of_month",      # 0 – 4  (floor((day-1)/7))
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "lat_diff",           # delivery_lat − pickup_lat
    "lon_diff",           # delivery_lon − pickup_lon
    "equipment_enc",      # label-encoded equipment type
    "pickup_enc",         # label-encoded pickup city
    "delivery_enc",       # label-encoded delivery city
    "lane_enc",           # label-encoded "pickup -> delivery" lane
]


# ─── Helper: Haversine distance ───────────────────────────────────────────────

def haversine_distance(
    lat1: np.ndarray | pd.Series,
    lon1: np.ndarray | pd.Series,
    lat2: np.ndarray | pd.Series,
    lon2: np.ndarray | pd.Series,
) -> np.ndarray:
    """Vectorised haversine great-circle distance in miles."""
    lat1, lon1, lat2, lon2 = map(np.asarray, (lat1, lon1, lat2, lon2))
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi   = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


# ─── Helper: Weight cleaning ──────────────────────────────────────────────────

def _clean_weight(
    df: pd.DataFrame,
    weight_medians: dict[str, float] | None = None,
    fit: bool = True,
) -> tuple[pd.Series, dict[str, float]]:
    """
    Fix weight sign errors (negative → abs) and impute missing values.

    Parameters
    ----------
    df             : dataframe containing 'weight' and 'equipment' columns
    weight_medians : per-equipment medians computed on training fold
                     (required when fit=False)
    fit            : if True, compute medians from df; else use weight_medians

    Returns
    -------
    w              : cleaned weight series (same index as df)
    weight_medians : dict {equipment -> median weight}
    """
    w = df["weight"].copy().abs()     # fix sign errors

    if fit:
        weight_medians = {}
        for eq in df["equipment"].unique():
            mask = df["equipment"] == eq
            weight_medians[eq] = float(w[mask].median())

    # Impute missing per equipment
    missing = w.isna()
    if missing.any():
        for eq, med in (weight_medians or {}).items():
            mask = (df["equipment"] == eq) & missing
            w.loc[mask] = med
        # global fallback
        global_med = float(np.nanmedian(list((weight_medians or {}).values()) or [w.median()]))
        w = w.fillna(global_med)

    return w, weight_medians or {}


# ─── Helper: Market index fill ────────────────────────────────────────────────

def _fill_market_index(
    df: pd.DataFrame,
    date_col: pd.Series,            # already-parsed datetime series
    global_mi_median: float | None = None,
) -> tuple[pd.Series, float]:
    """
    Fill missing market_index with that date's median, computed from the
    non-missing rows in *this* dataframe only (no cross-fold leakage).

    Falls back to the provided global_mi_median (computed on training fold)
    for dates that have no observed value in df at all.

    Returns
    -------
    mi_filled       : fully filled market_index series
    global_median   : global median computed from df (training fold value)
    """
    mi = df["market_index"].copy()

    # Per-date medians from non-missing rows in this df
    date_medians: dict = (
        pd.DataFrame({"date": date_col, "mi": mi})
        .dropna(subset=["mi"])
        .groupby("date")["mi"]
        .median()
        .to_dict()
    )

    missing = mi.isna()
    if missing.any():
        mi.loc[missing] = date_col[missing].map(date_medians)

    # Compute global median from this df for use as fallback
    df_global_median = float(np.nanmedian(list(date_medians.values()))) if date_medians else np.nan

    # Still-missing after date fill → use provided or df global median
    if mi.isna().any():
        fallback = global_mi_median if global_mi_median is not None else df_global_median
        mi = mi.fillna(fallback)

    return mi, df_global_median


# ─── Main: Build feature matrix ───────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    fit: bool = True,
    artifacts: dict | None = None,
) -> tuple[np.ndarray, list[str], dict]:
    """
    Build the feature matrix from a raw freight dataframe.

    Parameters
    ----------
    df        : raw dataframe (train-test or validation)
    fit       : True → fit encoders/maps from df (training mode)
                False → apply pre-fitted artifacts (inference mode)
    artifacts : dict produced by a previous call with fit=True.
                Required when fit=False.
                Keys: 'encoders', 'weight_medians', 'global_mi_median'

    Returns
    -------
    X             : float64 array, shape (n, len(FEATURE_NAMES))
    FEATURE_NAMES : list of feature column names
    artifacts     : fitted artifacts dict (reuse for inference)
    """
    df = df.copy()
    date_col = pd.to_datetime(df["date"])

    # Unpack / initialise artifacts
    if not fit:
        assert artifacts is not None, "artifacts must be provided when fit=False"
        encoders       = artifacts["encoders"]
        weight_medians = artifacts["weight_medians"]
        global_mi_med  = artifacts["global_mi_median"]
    else:
        encoders = {}
        weight_medians = None
        global_mi_med  = None

    # ── Weight ────────────────────────────────────────────────────────
    df["weight_clean"], weight_medians = _clean_weight(
        df, weight_medians=weight_medians, fit=fit
    )

    # ── Market index ──────────────────────────────────────────────────
    df["market_index_filled"], computed_global_mi = _fill_market_index(
        df, date_col, global_mi_median=global_mi_med
    )
    if fit:
        global_mi_med = computed_global_mi

    # ── Haversine & distance features ─────────────────────────────────
    df["hav_dist"] = haversine_distance(
        df["pickup_lat"], df["pickup_lon"],
        df["delivery_lat"], df["delivery_lon"],
    )
    df["dist_minus_hav"] = df["distance"] - df["hav_dist"]

    # ── Date features (NO month_of_year, NO day_of_year) ──────────────
    # Trees cannot extrapolate monotone time features to unseen months;
    # day-of-week and day-of-month capture cyclical patterns safely.
    df["day_of_week"]  = date_col.dt.dayofweek          # 0=Mon … 6=Sun
    df["day_of_month"] = date_col.dt.day                  # 1–31
    df["is_weekend"]   = (date_col.dt.dayofweek >= 5).astype(int)
    df["week_of_month"]= (date_col.dt.day - 1) // 7      # 0–4

    # ── Quote signal features ─────────────────────────────────────────
    df["quote_x_dist"] = df["quote_signal"] * df["distance"]
    df["quote_x_hav"]  = df["quote_signal"] * df["hav_dist"]

    # ── Geographic diffs ──────────────────────────────────────────────
    df["lat_diff"] = df["delivery_lat"] - df["pickup_lat"]
    df["lon_diff"] = df["delivery_lon"] - df["pickup_lon"]

    # ── Lane string ───────────────────────────────────────────────────
    df["lane"] = df["pickup"].astype(str) + " -> " + df["delivery"].astype(str)

    # ── Label encoding ────────────────────────────────────────────────
    cat_cols = ["equipment", "pickup", "delivery", "lane"]
    for col in cat_cols:
        if fit:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            # Handle unseen labels gracefully by mapping to a dedicated class
            known = set(le.classes_)
            mapped = df[col].astype(str).apply(
                lambda x: x if x in known else "__unknown__"
            )
            if "__unknown__" not in known:
                # Extend classes array without re-fitting
                le.classes_ = np.append(le.classes_, "__unknown__")
            df[col + "_enc"] = le.transform(mapped)

    # ── Assemble feature matrix ───────────────────────────────────────
    X = df[FEATURE_NAMES].values.astype(np.float64)

    fitted_artifacts = {
        "encoders": encoders,
        "weight_medians": weight_medians,
        "global_mi_median": global_mi_med,
    }

    return X, FEATURE_NAMES, fitted_artifacts


# ─── Utility: outlier mask ────────────────────────────────────────────────────

def rpm_outlier_mask(posted_rate: pd.Series, distance: pd.Series) -> pd.Series:
    """
    Return a boolean Series: True where rate-per-mile is an outlier.
    Thresholds: RPM < 1.2 or RPM > 4.5 (per assessment spec).
    """
    rpm = posted_rate / distance
    return (rpm < RPM_LOW_OUTLIER) | (rpm > RPM_HIGH_OUTLIER)
