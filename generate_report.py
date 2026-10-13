"""
generate_report.py — Create report.pdf from CV results and December chart.

Run from repository root:
    python generate_report.py
"""

from __future__ import annotations
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

ROOT    = Path(__file__).resolve().parent
CV_PATH = ROOT / "outputs" / "cv_results.csv"
CHART   = ROOT / "scorer_results" / "candidate_december.png"
OUT_PDF = ROOT / "report.pdf"

PRIMARY = "#064A56"
ACCENT  = "#1A8A9E"
LIGHT   = "#EFF6F7"
MID     = "#9DAFB3"
TEXT    = "#1C2C30"

def title_page(pdf: PdfPages) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.set_facecolor(PRIMARY)
    fig.patch.set_facecolor(PRIMARY)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.85, "Freight Rate Prediction", ha="center", va="center",
            fontsize=28, fontweight="bold", color="white",
            transform=ax.transAxes)
    ax.text(0.5, 0.78, "Machine Learning Assessment — Solution Report", ha="center",
            va="center", fontsize=15, color=MID, transform=ax.transAxes)

    lines = [
        ("Objective",   "Predict posted_rate for 12,000 loads in validation.csv"),
        ("Training data", "48,000 labeled loads  (Jan–Oct 2025)"),
        ("Validation data", "12,000 unlabeled loads  (Nov–Dec 2025)"),
        ("Model",       "HistGradientBoostingRegressor  |  sklearn"),
        ("Target",      "log(rate-per-mile)  →  prediction = exp(output) × distance"),
        ("CV strategy", "3-fold rolling-origin (expanding window)"),
        ("Seed",        "42 (fully deterministic)"),
    ]
    y0 = 0.65
    for label, value in lines:
        ax.text(0.12, y0, f"{label}:", ha="left", va="center",
                fontsize=10, color=MID, fontweight="bold", transform=ax.transAxes)
        ax.text(0.38, y0, value, ha="left", va="center",
                fontsize=10, color="white", transform=ax.transAxes)
        y0 -= 0.055

    ax.text(0.5, 0.10, "Spotter  ·  Freight Rate ML Assessment", ha="center",
            fontsize=9, color=MID, transform=ax.transAxes)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def data_quality_page(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111)
    ax.axis("off")

    def section(y, title):
        ax.text(0.0, y, title, fontsize=13, fontweight="bold", color=PRIMARY,
                transform=ax.transAxes, va="top")
        ax.plot([0, 1], [y - 0.018, y - 0.018], color=ACCENT, linewidth=1.5,
                transform=ax.transAxes, clip_on=False)

    # ── Page title ────────────────────────────────────────────────────
    ax.text(0.0, 0.97, "Section 1 · Data Quality & Validation Design",
            fontsize=16, fontweight="bold", color=TEXT, transform=ax.transAxes, va="top")

    # ── Data quality table ────────────────────────────────────────────
    section(0.88, "1.1  Data Quality Issues")

    col_labels = ["Issue", "Train count", "Val count", "Treatment"]
    rows_data = [
        ["Weight < 0 (sign error)",    "292",  "145",  "abs()"],
        ["Weight missing",             "300",  "165",  "Median by equipment"],
        ["Weight right-censored ≥47,500", "1,191", "—", "Kept as-is (informative ceiling)"],
        ["market_index missing",       "374",  "249",  "Per-date median from inputs only"],
        ["quote_signal missing",       "0",    "0",    "—"],
        ["RPM outliers <1.2 $/mi",     "335",  "—",    "Excluded from training only"],
        ["RPM outliers >4.5 $/mi",     "335",  "—",    "Excluded from training only"],
        ["Duplicate load_ids",         "0",    "0",    "—"],
        ["New cities in validation",   "—",    "8",    "Lat/lon + lane encoding handle OOV"],
        ["Distance vs haversine diff", "Mean +170 mi", "—", "Both used as features; road distance is routing distance"],
    ]

    table = ax.table(
        cellText=rows_data,
        colLabels=col_labels,
        loc="center",
        bbox=[0.0, 0.52, 1.0, 0.33],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor(PRIMARY)
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor(LIGHT)
        cell.set_edgecolor(MID)
        cell.set_linewidth(0.5)

    # ── Validation design ─────────────────────────────────────────────
    section(0.50, "1.2  Validation Design")

    design_text = (
        "Time-based rolling-origin CV (expanding window) with 3 folds:\n\n"
        "  Fold 1:  Train Jan–Jun 2025  |  Validate Jul–Aug 2025   (n_train≈28,806  n_val≈9,671)\n"
        "  Fold 2:  Train Jan–Jul 2025  |  Validate Aug–Sep 2025   (n_train≈33,718  n_val≈9,429)\n"
        "  Fold 3:  Train Jan–Aug 2025  |  Validate Sep–Oct 2025   (n_train≈38,477  n_val≈9,523)\n\n"
        "Rationale: The validation set (Nov–Dec) is temporally ahead of training. Random k-fold\n"
        "would constitute future leakage. Rolling-origin mirrors the real production scenario\n"
        "where we train on historical data and predict a future window.\n\n"
        "month_of_year and day_of_year are deliberately excluded: gradient-boosted trees\n"
        "cannot extrapolate a monotone trend to unseen months (Nov–Dec). Instead we use\n"
        "day_of_week, day_of_month, and market_index (which captures macro-trend)."
    )
    ax.text(0.0, 0.47, design_text, transform=ax.transAxes, va="top",
            fontsize=8.5, color=TEXT, family="monospace",
            bbox=dict(facecolor=LIGHT, edgecolor=MID, boxstyle="round,pad=0.4"))

    # ── City coverage note ────────────────────────────────────────────
    section(0.22, "1.3  City Coverage")
    city_text = (
        "8 cities appear in validation.csv that are absent from train-test.csv:\n"
        "  Norfolk, Allentown, Laredo, Chicago, Knoxville, San Diego, Jackson, Charlotte\n\n"
        "These are handled via:\n"
        "  • pickup_enc / delivery_enc : unseen labels mapped to '__unknown__' class\n"
        "  • lane_enc                  : same OOV mapping\n"
        "  • Geographic features (lat/lon, lat_diff, lon_diff, hav_dist) still fully informative"
    )
    ax.text(0.0, 0.19, city_text, transform=ax.transAxes, va="top",
            fontsize=8.5, color=TEXT, family="monospace",
            bbox=dict(facecolor=LIGHT, edgecolor=MID, boxstyle="round,pad=0.4"))

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def cv_results_page(pdf: PdfPages) -> None:
    cv = pd.read_csv(CV_PATH)
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111)
    ax.axis("off")

    ax.text(0.0, 0.97, "Section 2 · Model Reasoning & CV Results",
            fontsize=16, fontweight="bold", color=TEXT, transform=ax.transAxes, va="top")

    # ── Model reasoning ───────────────────────────────────────────────
    ax.text(0.0, 0.90, "2.1  Model Reasoning", fontsize=13, fontweight="bold",
            color=PRIMARY, transform=ax.transAxes, va="top")
    ax.plot([0, 1], [0.876, 0.876], color=ACCENT, linewidth=1.5, transform=ax.transAxes, clip_on=False)

    reasoning = (
        "Target: log(RPM)  where  RPM = posted_rate / distance\n"
        "Prediction: exp(model_output) × distance\n\n"
        "Why log(RPM)?\n"
        "  • Rate-per-mile distribution is right-skewed; log stabilises variance\n"
        "  • Multiplicative structure matches how freight prices actually work\n"
        "  • Distance factor separates lane-length effect from price-per-mile signal\n\n"
        "Model: sklearn HistGradientBoostingRegressor\n"
        "  • Handles missing values natively (NaN = its own split direction)\n"
        "  • Fast, memory-efficient on 48k rows\n"
        "  • No boosting libraries (LightGBM/XGBoost) per spec\n\n"
        "Comparison: squared_error vs absolute_error loss (both with log-RPM target)\n"
        "  → squared_error selected: CV avg non-outlier WAPE 3.34% vs 3.38% for absolute_error\n\n"
        "Features (22 total):\n"
        "  distance, hav_dist, dist_minus_hav, weight_clean, market_index_filled,\n"
        "  quote_signal, quote×dist, quote×hav, day_of_week, day_of_month,\n"
        "  is_weekend, week_of_month, pickup_lat, pickup_lon, delivery_lat,\n"
        "  delivery_lon, lat_diff, lon_diff, equipment_enc, pickup_enc,\n"
        "  delivery_enc, lane_enc"
    )
    ax.text(0.0, 0.855, reasoning, transform=ax.transAxes, va="top",
            fontsize=8.2, color=TEXT, family="monospace",
            bbox=dict(facecolor=LIGHT, edgecolor=MID, boxstyle="round,pad=0.4"))

    # ── CV table ──────────────────────────────────────────────────────
    ax.text(0.0, 0.46, "2.2  CV Results", fontsize=13, fontweight="bold",
            color=PRIMARY, transform=ax.transAxes, va="top")
    ax.plot([0, 1], [0.446, 0.446], color=ACCENT, linewidth=1.5, transform=ax.transAxes, clip_on=False)

    # Build a summary table: avg metrics, all + noo subsets
    avg = cv[cv["fold"] == "avg"].copy()
    rows_data, col_labels = [], []

    col_labels = ["Model", "Subset", "MAE ($)", "RMSE ($)", "MAPE (%)", "MedAPE (%)", "WAPE (%)"]
    model_names = {"naive": "Naive baseline", "hgbr_sq": "HGBR/sq_err ✓", "hgbr_ae": "HGBR/abs_err"}
    subset_names = {"all": "All rows", "noo": "Non-outlier"}

    for model in ["naive", "hgbr_sq", "hgbr_ae"]:
        for subset in ["all", "noo"]:
            r = avg[(avg["model"] == model) & (avg["subset"] == subset)]
            if r.empty:
                continue
            r = r.iloc[0]
            rows_data.append([
                model_names[model],
                subset_names[subset],
                f"{r['mae']:,.1f}",
                f"{r['rmse']:,.1f}",
                f"{r['mape']:.2f}",
                f"{r['med_ape']:.2f}",
                f"{r['wape']:.2f}",
            ])

    # Also per-fold detail (fold 1-3, hgbr_sq, noo only)
    folds_detail = cv[(cv["fold"].astype(str).isin(["1", "2", "3"])) &
                      (cv["model"] == "hgbr_sq") &
                      (cv["subset"] == "noo")].copy()

    table = ax.table(
        cellText=rows_data,
        colLabels=col_labels,
        loc="center",
        bbox=[0.0, 0.22, 1.0, 0.21],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor(PRIMARY)
            cell.set_text_props(color="white", fontweight="bold")
        elif rows_data[r - 1][0] == "HGBR/sq_err ✓":
            cell.set_facecolor("#D4EDF0")
        elif r % 2 == 0:
            cell.set_facecolor(LIGHT)
        cell.set_edgecolor(MID)
        cell.set_linewidth(0.5)

    # Per-fold detail (HGBR/sq, non-outlier)
    ax.text(0.0, 0.21, "Per-fold detail — HGBR/squared_error, non-outlier rows:",
            fontsize=8, color=TEXT, transform=ax.transAxes, va="top", style="italic")

    detail_rows = []
    for _, r in folds_detail.iterrows():
        detail_rows.append([
            f"Fold {r['fold']}  ({r['val_start']}–{r['val_end']})",
            f"n={r['n']:,}",
            f"{r['mae']:,.1f}",
            f"{r['rmse']:,.1f}",
            f"{r['mape']:.2f}%",
            f"{r['med_ape']:.2f}%",
            f"{r['wape']:.2f}%",
        ])

    table2 = ax.table(
        cellText=detail_rows,
        colLabels=["Fold (val range)", "n", "MAE ($)", "RMSE ($)", "MAPE", "MedAPE", "WAPE"],
        loc="center",
        bbox=[0.0, 0.02, 1.0, 0.17],
    )
    table2.auto_set_font_size(False)
    table2.set_fontsize(7.5)
    for (r, c), cell in table2.get_celld().items():
        if r == 0:
            cell.set_facecolor(ACCENT)
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor(LIGHT)
        cell.set_edgecolor(MID)
        cell.set_linewidth(0.5)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def chart_and_caveats_page(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.45,
                           top=0.93, bottom=0.05, left=0.06, right=0.96)

    # ── Page title ────────────────────────────────────────────────────
    fig.text(0.03, 0.96,
             "Section 3 · December Chart  &  Caveats",
             fontsize=16, fontweight="bold", color=TEXT, va="top")

    # ── December chart ────────────────────────────────────────────────
    ax_chart = fig.add_subplot(gs[0:2, 0])
    chart_img = plt.imread(CHART)
    ax_chart.imshow(chart_img)
    ax_chart.axis("off")
    ax_chart.set_title("December 2025 Predicted Load Rate  (Lexington → Fort Wayne, 360 mi, Dry Van)",
                        fontsize=9, color=TEXT, pad=4)

    # ── Caveats ───────────────────────────────────────────────────────
    ax_cav = fig.add_subplot(gs[2, 0])
    ax_cav.axis("off")
    ax_cav.set_title("Caveats & Assumptions", fontsize=11, fontweight="bold",
                     color=PRIMARY, loc="left", pad=6)

    caveats = (
        "1. December market_index and quote_signal are taken as the per-date median of\n"
        "   validation.csv inputs for December 2025. These are market-observed values;\n"
        "   their medians are a reasonable point estimate when a single-row prediction is required.\n\n"
        "2. Lexington → Fort Wayne is a 360-mile route. The haversine distance between the\n"
        "   stored coordinates is ~284 mi; the 76-mi gap reflects routing detours and is\n"
        "   consistent with training data (avg road-haversine gap = 170 mi).\n\n"
        "3. 8 cities appear in validation.csv but not in train-test.csv. These are handled\n"
        "   via an OOV '__unknown__' bucket in label encoders; geographic features\n"
        "   (lat/lon, diffs, haversine) remain fully informative for these lanes.\n\n"
        "4. The model was trained on Jan–Oct 2025 data. Nov–Dec 2025 freight market\n"
        "   conditions (e.g., holiday surges) are extrapolated from seasonal signals\n"
        "   captured by day_of_week, day_of_month, and market_index. Structural breaks\n"
        "   (strikes, weather events) cannot be predicted from historical patterns alone.\n\n"
        "5. RPM outliers (< $1.20/mi or > $4.50/mi) were excluded from training (670 rows,\n"
        "   1.4% of data) per assessment spec. They are included in 'all rows' evaluation."
    )
    ax_cav.text(0.0, 0.98, caveats, transform=ax_cav.transAxes, va="top",
                fontsize=7.8, color=TEXT,
                bbox=dict(facecolor=LIGHT, edgecolor=MID, boxstyle="round,pad=0.5"))

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print(f"Generating report → {OUT_PDF}")
    with PdfPages(OUT_PDF) as pdf:
        title_page(pdf)
        data_quality_page(pdf)
        cv_results_page(pdf)
        chart_and_caveats_page(pdf)
    print("Done.")


if __name__ == "__main__":
    main()
