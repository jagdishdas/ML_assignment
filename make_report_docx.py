"""
make_report_docx.py  — Generate report.docx (US Letter, ~2-3 pages)

Model numbers are read from the result files (nothing is typed in by hand):
  outputs/cv_results.csv
  outputs/ablation_results.csv
  december_chart_inputs_completed.csv
  scorer_results/candidate_december.png
Data-quality counts in Section 2 and the split sizes in Section 3 were
computed from the raw CSVs and are fixed text.

Run from repo root:
    python make_report_docx.py
"""
from __future__ import annotations
import io
from pathlib import Path

import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL

ROOT      = Path(__file__).resolve().parent
CV_PATH   = ROOT / "outputs" / "cv_results.csv"
ABL_PATH  = ROOT / "outputs" / "ablation_results.csv"
DEC_PATH  = ROOT / "december_chart_inputs_completed.csv"
CHART_IMG = ROOT / "scorer_results" / "candidate_december.png"
OUT_DOCX  = ROOT / "report.docx"

# ── Results loaded from files ─────────────────────────────────────────────────
CV  = pd.read_csv(CV_PATH)
ABL = pd.read_csv(ABL_PATH)
_A0 = ABL.iloc[0]                                        # original 22 features
_AF = ABL[ABL["config"].str.contains(r"\[final\]")].iloc[0]  # final configuration
DEC = pd.read_csv(DEC_PATH)

_AVG = CV[CV["fold"] == "avg"].set_index(["model", "subset"])
_FOLD = CV[CV["fold"] != "avg"].copy()
_FOLD["fold"] = _FOLD["fold"].astype(int)
# Selected model = lower average non-outlier WAPE (same rule as src/train.py)
SEL = ("hgbr_sq" if _AVG.loc[("hgbr_sq", "noo"), "wape"] <= _AVG.loc[("hgbr_ae", "noo"), "wape"]
       else "hgbr_ae")
OTHER = "hgbr_ae" if SEL == "hgbr_sq" else "hgbr_sq"
LOSS_NAME = {"hgbr_sq": "squared_error", "hgbr_ae": "absolute_error"}


def avg(model: str, subset: str, col: str = "wape") -> float:
    return float(_AVG.loc[(model, subset), col])


def red(subset: str) -> int:
    """% reduction in WAPE of the selected model vs the naive baseline."""
    return round(100 * (1 - avg(SEL, subset) / avg("naive", subset)))


_dec_dates = pd.to_datetime(DEC["date"])
DEC_MIN, DEC_MAX, DEC_MEAN = DEC["predicted_rate"].min(), DEC["predicted_rate"].max(), DEC["predicted_rate"].mean()
_dow = DEC.groupby(_dec_dates.dt.dayofweek)["predicted_rate"].mean()
_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# ── Brand colours ─────────────────────────────────────────────────────────────
TEAL_DARK  = RGBColor(0x06, 0x4A, 0x56)   # #064A56
TEAL_MED   = RGBColor(0x1A, 0x8A, 0x9E)   # #1A8A9E
TEAL_LIGHT = RGBColor(0xD4, 0xED, 0xF0)   # #D4EDF0  (header fill)
WHITE      = RGBColor(0xFF, 0xFF, 0xFF)
GREY_TEXT  = RGBColor(0x44, 0x44, 0x44)

FONT_NAME  = "Calibri"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _set_cell_bg(cell, hex_color: str) -> None:
    """Fill a table cell with a solid background colour (hex without #)."""
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd  = OxmlElement("w:shd")
    shd.set(qn("w:val"),   "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"),  hex_color)
    tcPr.append(shd)


def _cell_text(cell, text: str, bold=False, font_size=9,
               color: RGBColor | None = None,
               align=WD_ALIGN_PARAGRAPH.LEFT) -> None:
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    p   = cell.paragraphs[0]
    p.alignment = align
    run = p.add_run(text)
    run.bold = bold
    run.font.name = FONT_NAME
    run.font.size = Pt(font_size)
    if color:
        run.font.color.rgb = color


def _heading(doc: Document, text: str, level: int = 1) -> None:
    p = doc.add_paragraph(style=f"Heading {level}")
    p.clear()
    run = p.add_run(text)
    run.font.name = FONT_NAME
    run.font.bold = True
    run.font.color.rgb = TEAL_DARK
    run.font.size = Pt(13 if level == 1 else 11)
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after  = Pt(3)


def _body(doc: Document, text: str, space_after: int = 4) -> None:
    p = doc.add_paragraph(text)
    p.style = "Normal"
    for run in p.runs:
        run.font.name = FONT_NAME
        run.font.size = Pt(10)
        run.font.color.rgb = GREY_TEXT
    p.paragraph_format.space_after = Pt(space_after)


def _bold_inline(para, text: str, suffix: str = "") -> None:
    r = para.add_run(text)
    r.bold = True
    r.font.name = FONT_NAME
    r.font.size = Pt(10)
    r.font.color.rgb = TEAL_DARK
    if suffix:
        r2 = para.add_run(suffix)
        r2.font.name = FONT_NAME
        r2.font.size = Pt(10)
        r2.font.color.rgb = GREY_TEXT


def _add_horizontal_rule(doc: Document) -> None:
    p = doc.add_paragraph()
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"),   "single")
    bottom.set(qn("w:sz"),    "6")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "1A8A9E")
    pBdr.append(bottom)
    pPr.append(pBdr)
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after  = Pt(2)


# ── Document setup ────────────────────────────────────────────────────────────

def setup_document() -> Document:
    doc = Document()
    # US Letter = 8.5 × 11 in
    section = doc.sections[0]
    section.page_width  = Inches(8.5)
    section.page_height = Inches(11)
    section.left_margin   = Inches(1.0)
    section.right_margin  = Inches(1.0)
    section.top_margin    = Inches(0.9)
    section.bottom_margin = Inches(0.9)
    # Default font
    doc.styles["Normal"].font.name = FONT_NAME
    doc.styles["Normal"].font.size = Pt(10)
    return doc


# ── Title block ───────────────────────────────────────────────────────────────

def add_title(doc: Document) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = p.add_run("Freight Rate Prediction — Solution Report")
    run.font.name  = FONT_NAME
    run.font.size  = Pt(18)
    run.font.bold  = True
    run.font.color.rgb = TEAL_DARK
    p.paragraph_format.space_after = Pt(1)

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r2  = sub.add_run(
        "Machine Learning Engineer Assessment  ·  "
        "HistGradientBoostingRegressor on log(rate-per-mile)  ·  seed = 42"
    )
    r2.font.name  = FONT_NAME
    r2.font.size  = Pt(9.5)
    r2.font.color.rgb = RGBColor(0x45, 0x5A, 0x60)
    sub.paragraph_format.space_after = Pt(6)
    _add_horizontal_rule(doc)


# ── Section 1: Summary ────────────────────────────────────────────────────────

def add_summary(doc: Document) -> None:
    _heading(doc, "1. Summary")
    _body(doc,
        "The goal is to predict posted_rate (freight load price in USD) for 12,000 "
        "unseen loads covering November–December 2025, using 48,000 labeled loads "
        "(January–October 2025) for training. The model is a gradient-boosted "
        "tree (sklearn HistGradientBoostingRegressor) that learns log(rate-per-mile); "
        "the dollar prediction is recovered as exp(output) × distance. "
        "Cross-validation uses three rolling-origin (expanding-window) folds, "
        "matching the temporal structure of the holdout set. "
        f"The best model achieves an average non-outlier WAPE of {avg(SEL, 'noo'):.2f}% versus "
        f"{avg('naive', 'noo'):.2f}% for the naive median-rate-per-mile baseline on the same rows "
        f"(a {red('noo')}% reduction in weighted absolute error); including the outlier "
        f"rows the figures are {avg(SEL, 'all'):.2f}% versus {avg('naive', 'all'):.2f}% "
        f"(a {red('all')}% reduction). "
        "These are cross-validation figures on January–October only; the "
        "November–December validation labels are not available."
    )


# ── Section 2: Data Quality ───────────────────────────────────────────────────

def add_data_quality(doc: Document) -> None:
    _heading(doc, "2. Data and Quality Issues")

    # Intro sentence
    _body(doc,
        "data/train_test.csv contains 48,000 rows (Jan–Oct 2025, 14 columns). "
        "data/validation.csv contains 12,000 rows (Nov–Dec 2025, 13 columns — no posted_rate). "
        "The following issues were identified and treated before feature engineering:",
        space_after=5
    )

    # Table: issue / scale / treatment
    headers = ["Issue", "Scale (train / val)", "Treatment"]
    rows = [
        ["Weight negative (sign error)",
         "292 train  /  145 val",
         "abs() — take absolute value"],
        ["Weight missing",
         "300 train  /  165 val",
         "Median by equipment type, then global median"],
        ["Weight right-censored ≥ 47,500 lb",
         "1,191 train  /  297 val",
         "Kept as-is; ceiling value is informative"],
        ["market_index missing",
         "374 train  /  249 val",
         "Per-date median from same dataset's non-missing rows only (no leakage)"],
        ["quote_signal missing",
         "0 train  /  0 val",
         "No action required"],
        ["RPM outliers  < $1.20/mi or > $4.50/mi",
         "335 + 335 = 670 train  /  —",
         "Excluded from training only; kept in 'all rows' evaluation"],
        ["Duplicate load_ids",
         "0 train  /  0 val",
         "No action required"],
        ["Cities in val absent from train",
         "8 cities (e.g. Chicago, Laredo)",
         "Mapped to an '__unknown__' code (never seen in training, so not learned); lat/lon features still describe the city"],
        ["road distance vs haversine",
         "Mean gap = +170 mi (road >> straight-line)",
         "Both distance and hav_dist used as features; diff = routing detour proxy"],
    ]

    tbl = doc.add_table(rows=len(rows) + 1, cols=3)
    tbl.alignment = WD_TABLE_ALIGNMENT.LEFT
    tbl.style     = "Table Grid"

    col_widths = [Inches(1.9), Inches(1.85), Inches(2.65)]
    for i, width in enumerate(col_widths):
        for row in tbl.rows:
            row.cells[i].width = width

    # Header
    for j, h in enumerate(headers):
        _set_cell_bg(tbl.rows[0].cells[j], "064A56")
        _cell_text(tbl.rows[0].cells[j], h, bold=True,
                   font_size=9, color=WHITE,
                   align=WD_ALIGN_PARAGRAPH.CENTER)

    # Data rows
    for i, row_data in enumerate(rows, 1):
        bg = "D4EDF0" if i % 2 == 0 else "FFFFFF"
        for j, val in enumerate(row_data):
            _set_cell_bg(tbl.rows[i].cells[j], bg)
            _cell_text(tbl.rows[i].cells[j], val, font_size=8.5)

    doc.add_paragraph().paragraph_format.space_after = Pt(2)


# ── Section 3: Train / Validation Split ──────────────────────────────────────

def add_split(doc: Document) -> None:
    _heading(doc, "3. Train / Validation Split")

    p = doc.add_paragraph()
    _bold_inline(p, "Why random K-fold is wrong here:  ",
                 "The validation set (Nov–Dec 2025) is strictly later than the "
                 "training set (Jan–Oct 2025). Random K-fold would mix future "
                 "labels into earlier training folds, artificially inflating "
                 "performance estimates, and it would not mimic the real task of "
                 "predicting a later period (Nov–Dec) from earlier data only.")
    p.paragraph_format.space_after = Pt(4)

    p2 = doc.add_paragraph()
    _bold_inline(p2, "Strategy used — rolling-origin (expanding window):  ",
                 "Three forward-chaining folds were used. In each fold the "
                 "training window expands by one month and the two-month validation "
                 "window advances by one month (adjacent validation windows "
                 "overlap by one month), mirroring production conditions.")
    p2.paragraph_format.space_after = Pt(5)

    # Fold table
    f_headers = ["Fold", "Training window", "Validation window", "n train", "n val"]
    f_rows = [
        ["1", "Jan 2025 – Jun 2025", "Jul 2025 – Aug 2025", "28,806", "9,671"],
        ["2", "Jan 2025 – Jul 2025", "Aug 2025 – Sep 2025", "33,718", "9,429"],
        ["3", "Jan 2025 – Aug 2025", "Sep 2025 – Oct 2025", "38,477", "9,523"],
    ]

    tbl = doc.add_table(rows=4, cols=5)
    tbl.style = "Table Grid"
    tbl.alignment = WD_TABLE_ALIGNMENT.LEFT

    for j, h in enumerate(f_headers):
        _set_cell_bg(tbl.rows[0].cells[j], "1A8A9E")
        _cell_text(tbl.rows[0].cells[j], h, bold=True,
                   font_size=9, color=WHITE,
                   align=WD_ALIGN_PARAGRAPH.CENTER)

    for i, row_data in enumerate(f_rows, 1):
        bg = "EFF6F7" if i % 2 == 0 else "FFFFFF"
        for j, val in enumerate(row_data):
            _set_cell_bg(tbl.rows[i].cells[j], bg)
            _cell_text(tbl.rows[i].cells[j], val, font_size=9,
                       align=WD_ALIGN_PARAGRAPH.CENTER)

    doc.add_paragraph().paragraph_format.space_after = Pt(2)


# ── Section 4: Model ──────────────────────────────────────────────────────────

def add_model(doc: Document) -> None:
    _heading(doc, "4. Model")

    p = doc.add_paragraph()
    _bold_inline(p, "Target — log(rate-per-mile):  ",
                 "Rate-per-mile (RPM = posted_rate / distance) is strongly right-skewed "
                 "(skewness 8.72 raw, −0.04 after the log). Modelling log(RPM) "
                 "gives a near-symmetric target, "
                 "ensures predictions are always positive, and respects the "
                 "multiplicative relationship between price-per-mile and distance. "
                 "Predictions are recovered as  exp(model output) × distance.")
    p.paragraph_format.space_after = Pt(4)

    p2 = doc.add_paragraph()
    _bold_inline(p2, "Algorithm — HistGradientBoostingRegressor (sklearn):  ",
                 "Gradient-boosted histogram trees handle the mixed feature space "
                 "(numeric distances, temporal day features, and equipment, city "
                 "and lane columns label-encoded as integers). Training rows are "
                 "weighted by recency, exp(−age/90 days), so the model follows the "
                 "latest market level. Missing weight and "
                 "market_index values are imputed before modelling, so the model "
                 "never sees NaNs. "
                 "max_iter=500, learning_rate=0.05, max_depth=6, "
                 "min_samples_leaf=20, l2_regularization=0.1, seed=42.")
    p2.paragraph_format.space_after = Pt(4)

    p3 = doc.add_paragraph()
    _bold_inline(p3, "What else was compared:  ",
                 "(a) Naive baseline: median RPM from non-outlier training rows × distance. "
                 "(b) HGBR with squared_error loss on log(RPM). "
                 "(c) HGBR with absolute_error loss on log(RPM). "
                 "Both HGBR variants far outperform the baseline. "
                 f"{LOSS_NAME[SEL]} was selected (avg non-outlier WAPE "
                 f"{avg(SEL, 'noo'):.2f}% vs {avg(OTHER, 'noo'):.2f}% for {LOSS_NAME[OTHER]}).")
    p3.paragraph_format.space_after = Pt(4)

    p4 = doc.add_paragraph()
    _bold_inline(p4, "Features (26 total):  ",
                 "distance, hav_dist, road−haversine gap, weight_clean, "
                 "market_index_filled, quote_signal, quote×distance, quote×haversine, "
                 "day_of_week, day_of_month, is_weekend, week_of_month, "
                 "pickup_lat/lon, delivery_lat/lon, lat_diff, lon_diff, "
                 "equipment_enc, pickup_enc, delivery_enc, lane_enc, plus four "
                 "market-context features: day_mi_median and day_qs_median (that date's "
                 "median market_index and quote_signal over all loads), quote_rel "
                 "(this load's quote_signal relative to its day's median) and day_index "
                 "(days since 2025-01-01). All use inputs only, never the target. "
                 "month_of_year is excluded; day_index is monotone, so every "
                 "Nov–Dec date falls into the latest training period rather than "
                 "being extrapolated.")
    p4.paragraph_format.space_after = Pt(4)

    p5 = doc.add_paragraph()
    _bold_inline(p5, "Why the market-context features:  ",
                 "market_index behaves like a daily market level (spread within a "
                 "day 0.025, across days 0.167), and after removing lane and equipment "
                 "effects it correlates 0.70 with that day's average price. With the "
                 "original 22 features the model under-predicted in every fold "
                 f"({_A0['bias_noo_fold1']:.1f}%, {_A0['bias_noo_fold2']:.1f}%, "
                 f"{_A0['bias_noo_fold3']:.1f}% mean signed error), a near-constant level "
                 "offset that a model with no time information cannot see. "
                 "Section 5 shows the effect of each addition.")
    p5.paragraph_format.space_after = Pt(2)


# ── Section 5: Results ────────────────────────────────────────────────────────

def add_results(doc: Document) -> None:
    _heading(doc, "5. Results")

    _body(doc,
        "3-fold rolling-origin CV averages. "
        "'All rows' includes the 670 RPM outliers; "
        "'Non-outlier' excludes rows where RPM < $1.20/mi or > $4.50/mi. "
        "Selected model marked ✓.",
        space_after=5
    )

    # Averaged results table
    col_h = ["Model", "Rows", "MAE ($)", "RMSE ($)", "MAPE (%)", "MedAPE (%)", "WAPE (%)"]
    data_rows = []
    for _m, _label in [("naive", "Naive baseline"),
                       ("hgbr_sq", "HGBR / squared_error"),
                       ("hgbr_ae", "HGBR / absolute_error")]:
        _label = _label + (" ✓" if _m == SEL else "")
        for _sub, _sublabel in [("all", "All rows"), ("noo", "Non-outlier")]:
            data_rows.append((_label, _sublabel,
                              f"{avg(_m, _sub, 'mae'):.2f}", f"{avg(_m, _sub, 'rmse'):.2f}",
                              f"{avg(_m, _sub, 'mape'):.2f}", f"{avg(_m, _sub, 'med_ape'):.2f}",
                              f"{avg(_m, _sub, 'wape'):.2f}"))

    tbl = doc.add_table(rows=len(data_rows) + 1, cols=7)
    tbl.style = "Table Grid"
    tbl.alignment = WD_TABLE_ALIGNMENT.LEFT

    col_widths_r = [Inches(1.65), Inches(0.85), Inches(0.65),
                    Inches(0.65), Inches(0.65), Inches(0.80), Inches(0.65)]
    for i, w in enumerate(col_widths_r):
        for row in tbl.rows:
            row.cells[i].width = w

    for j, h in enumerate(col_h):
        _set_cell_bg(tbl.rows[0].cells[j], "064A56")
        _cell_text(tbl.rows[0].cells[j], h, bold=True,
                   font_size=8.5, color=WHITE,
                   align=WD_ALIGN_PARAGRAPH.CENTER)

    for i, (model, subset, mae, rmse, mape, medape, wape) in enumerate(data_rows, 1):
        is_selected = "✓" in model
        if is_selected:
            bg = "C8E6C9"   # light green for selected model
        elif i % 2 == 0:
            bg = "EFF6F7"
        else:
            bg = "FFFFFF"

        for j, val in enumerate([model, subset, mae, rmse, mape, medape, wape]):
            _set_cell_bg(tbl.rows[i].cells[j], bg)
            _cell_text(tbl.rows[i].cells[j], val,
                       bold=is_selected, font_size=8.5,
                       align=WD_ALIGN_PARAGRAPH.CENTER if j > 1
                             else WD_ALIGN_PARAGRAPH.LEFT)

    # Per-fold detail
    doc.add_paragraph().paragraph_format.space_after = Pt(3)
    _body(doc, f"Per-fold breakdown — HGBR / {LOSS_NAME[SEL]} (selected model), non-outlier rows:", space_after=3)

    fold_col_h = ["Fold", "Val window",          "n val", "MAE ($)", "RMSE ($)", "MAPE (%)", "MedAPE (%)", "WAPE (%)"]
    _win = {1: "Jul–Aug 2025", 2: "Aug–Sep 2025", 3: "Sep–Oct 2025"}
    fold_rows = []
    for _f in (1, 2, 3):
        _r = _FOLD[(_FOLD["fold"] == _f) & (_FOLD["model"] == SEL) & (_FOLD["subset"] == "noo")].iloc[0]
        fold_rows.append((str(_f), _win[_f], f"{int(_r['n']):,}", f"{_r['mae']:.2f}",
                          f"{_r['rmse']:.2f}", f"{_r['mape']:.2f}", f"{_r['med_ape']:.2f}",
                          f"{_r['wape']:.2f}"))
    fold_rows.append(("Avg", "—", "—", f"{avg(SEL, 'noo', 'mae'):.2f}",
                      f"{avg(SEL, 'noo', 'rmse'):.2f}", f"{avg(SEL, 'noo', 'mape'):.2f}",
                      f"{avg(SEL, 'noo', 'med_ape'):.2f}", f"{avg(SEL, 'noo', 'wape'):.2f}"))

    tbl2 = doc.add_table(rows=len(fold_rows) + 1, cols=8)
    tbl2.style = "Table Grid"
    tbl2.alignment = WD_TABLE_ALIGNMENT.LEFT

    for j, h in enumerate(fold_col_h):
        _set_cell_bg(tbl2.rows[0].cells[j], "1A8A9E")
        _cell_text(tbl2.rows[0].cells[j], h, bold=True,
                   font_size=8.5, color=WHITE,
                   align=WD_ALIGN_PARAGRAPH.CENTER)

    for i, row_vals in enumerate(fold_rows, 1):
        bg = "EFF6F7" if i % 2 == 0 else "FFFFFF"
        if row_vals[0] == "Avg":
            bg = "D4EDF0"
        for j, val in enumerate(row_vals):
            _set_cell_bg(tbl2.rows[i].cells[j], bg)
            _cell_text(tbl2.rows[i].cells[j], val,
                       bold=(row_vals[0] == "Avg"),
                       font_size=8.5,
                       align=WD_ALIGN_PARAGRAPH.CENTER)

    doc.add_paragraph().paragraph_format.space_after = Pt(3)
    _body(doc,
        "Where the improvement comes from — one change at a time, same 3 folds, "
        "squared_error loss, non-outlier WAPE (%). Produced by src/ablation.py "
        "(outputs/ablation_results.csv). The last four rows vary the recency "
        "decay constant (days); 60–120 days gives nearly the same result, so the "
        "outcome does not hinge on the exact value.",
        space_after=3)

    abl_h = ["Configuration", "Fold 1", "Fold 2", "Fold 3", "Avg (non-outlier)", "Avg (all rows)"]
    tbl3 = doc.add_table(rows=len(ABL) + 1, cols=len(abl_h))
    tbl3.style = "Table Grid"
    tbl3.alignment = WD_TABLE_ALIGNMENT.LEFT
    for j, h in enumerate(abl_h):
        _set_cell_bg(tbl3.rows[0].cells[j], "1A8A9E")
        _cell_text(tbl3.rows[0].cells[j], h, bold=True, font_size=8.5,
                   color=WHITE, align=WD_ALIGN_PARAGRAPH.CENTER)
    for i, (_, a) in enumerate(ABL.iterrows(), 1):
        is_final = "[final]" in a["config"]
        bg = "C8E6C9" if is_final else ("EFF6F7" if i % 2 == 0 else "FFFFFF")
        vals = [a["config"], f"{a['wape_noo_fold1']:.2f}", f"{a['wape_noo_fold2']:.2f}",
                f"{a['wape_noo_fold3']:.2f}", f"{a['wape_noo_avg']:.2f}", f"{a['wape_all_avg']:.2f}"]
        for j, val in enumerate(vals):
            _set_cell_bg(tbl3.rows[i].cells[j], bg)
            _cell_text(tbl3.rows[i].cells[j], val, bold=is_final, font_size=8.5,
                       align=WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER)

    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    _body(doc,
        "Caveat: the recency constant and the feature set were chosen using these "
        "same three folds, so the final figure is somewhat optimistic. Adding the "
        "market features and day_index alone helps in folds 2–3 but not fold 1; "
        "the full final configuration beats the original in all three folds. "
        "The model still under-predicts "
        f"(mean signed error {_AF['bias_noo_fold1']:.1f}%, {_AF['bias_noo_fold2']:.1f}%, "
        f"{_AF['bias_noo_fold3']:.1f}% in folds 1–3, non-outlier rows).",
        space_after=2)


# ── Section 6: Fixed December Chart ──────────────────────────────────────────

def add_december(doc: Document) -> None:
    _heading(doc, "6. Fixed December Chart")

    p = doc.add_paragraph()
    _bold_inline(p, "Assumption — market_index and quote_signal:  ",
                 "The data/december_chart_inputs.csv file does not include these columns. "
                 "Both are filled with their per-date median from data/validation.csv inputs "
                 "for December 2025 (market_index median per day from the 6,030 non-missing "
                 "December validation rows; quote_signal median per day from all 6,164 "
                 "December validation rows). Lexington coordinates (36.99152, −84.99876) and "
                 "Fort Wayne coordinates (41.31561, −85.36206) are taken from data/train_test.csv. "
                 f"Predictions range ${DEC_MIN:,.2f}–${DEC_MAX:,.2f} (mean ${DEC_MEAN:,.2f}), "
                 f"a swing of about ${DEC_MAX - DEC_MIN:,.0f} ({(DEC_MAX - DEC_MIN) / DEC_MEAN * 100:.0f}%). "
                 f"By weekday the means are highest on {_DAYS[int(_dow.idxmax())]} "
                 f"(${_dow.max():,.1f}) and lowest on {_DAYS[int(_dow.idxmin())]} "
                 f"(${_dow.min():,.1f}).")
    p.paragraph_format.space_after = Pt(6)

    # Embed chart image
    doc.add_picture(str(CHART_IMG), width=Inches(6.3))
    last_para = doc.paragraphs[-1]
    last_para.alignment = WD_ALIGN_PARAGRAPH.CENTER

    caption = doc.add_paragraph(
        "Figure 1: December 2025 predicted load rate — Lexington → Fort Wayne, "
        "360 mi, Dry Van, 32,000 lb. Generated by score.py from "
        "december_chart_inputs_completed.csv."
    )
    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in caption.runs:
        run.font.size = Pt(8.5)
        run.font.color.rgb = RGBColor(0x45, 0x5A, 0x60)
        run.font.italic = True
    caption.paragraph_format.space_after = Pt(2)


# ── Section 7: Reproducibility ────────────────────────────────────────────────

def add_reproducibility(doc: Document) -> None:
    _heading(doc, "7. Reproducibility")

    _body(doc, "All commands are run from the ML_assignment/ repository root.", space_after=3)

    cmds = [
        ("# 1. Install dependencies",
         "pip install -r requirements.txt"),
        ("# 2. Train, cross-validate, and generate all prediction files",
         "python src/train.py"),
        ("# 3. Validate submissions and produce the December chart",
         "python score.py \\\n"
         "  --predictions validation_predictions.csv \\\n"
         "  --december-predictions december_chart_inputs_completed.csv \\\n"
         "  --output-dir scorer_results"),
        ("# 4. (Optional) Re-run the feature ablation (Section 5)",
         "python src/ablation.py"),
        ("# 5. (Optional) Regenerate this report",
         "python make_report_docx.py"),
    ]

    for comment, cmd in cmds:
        # Comment line in teal italic
        pc = doc.add_paragraph()
        rc = pc.add_run(comment)
        rc.font.name   = "Courier New"
        rc.font.size   = Pt(8.5)
        rc.font.italic = True
        rc.font.color.rgb = TEAL_MED
        pc.paragraph_format.space_before = Pt(4)
        pc.paragraph_format.space_after  = Pt(0)

        # Command block
        pb = doc.add_paragraph()
        rb = pb.add_run(cmd)
        rb.font.name  = "Courier New"
        rb.font.size  = Pt(8.5)
        rb.font.color.rgb = RGBColor(0x1C, 0x2C, 0x30)
        pb.paragraph_format.left_indent  = Inches(0.2)
        pb.paragraph_format.space_before = Pt(0)
        pb.paragraph_format.space_after  = Pt(1)

    doc.add_paragraph().paragraph_format.space_after = Pt(3)

    _body(doc,
        "Seed 42 is fixed in every call to HistGradientBoostingRegressor. "
        "Label encoders are fitted once on training data and reused for "
        "inference. Byte-for-byte identical outputs confirmed across two "
        "independent runs (max difference: 0.0).",
        space_after=2
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    doc = setup_document()
    add_title(doc)
    add_summary(doc)
    add_data_quality(doc)
    add_split(doc)
    add_model(doc)
    add_results(doc)
    add_december(doc)
    add_reproducibility(doc)

    doc.save(str(OUT_DOCX))
    print(f"Saved: {OUT_DOCX}")


if __name__ == "__main__":
    main()
