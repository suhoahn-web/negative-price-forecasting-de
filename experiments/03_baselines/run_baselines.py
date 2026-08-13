"""Baseline suite for negative-price episode forecasting, rolling origin.

Tasks
  A  y_neg  : negative price at hour t                     (hourly event)
  B  y_run4 : hour t is inside a negative run >= 4 h        (EEG-relevant)
  C  y_day4 : day D contains a negative run >= 4 h          (daily, the money question)

Models
  climatology (month x hour), persistence (D-2), logistic (calendar / +AR / +forecasts),
  LightGBM (same feature sets)

Evaluation: expanding-window rolling origin, one model per test year. Primary metric
PR-AUC (no-skill floor = base rate); ROC-AUC and Brier reported alongside because the
literature over-reports ROC-AUC at low base rates.

Outputs: outputs/tables/baseline_results.csv, experiments/03_baselines/BASELINE_NOTES.md
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.features import (FEATURES_AR, FEATURES_CAL, FEATURES_FC,  # noqa: E402
                          build, load_hourly)

TEST_YEARS = [2021, 2022, 2023, 2024, 2025]
SEED = 42
FEATURE_SETS = {
    "cal": FEATURES_CAL,
    "cal+AR": FEATURES_CAL + FEATURES_AR,
    "cal+AR+FC": FEATURES_CAL + FEATURES_AR + FEATURES_FC,
    "cal+FC": FEATURES_CAL + FEATURES_FC,
}


def score(y, p) -> dict:
    return {"PR_AUC": float(average_precision_score(y, p)),
            "ROC_AUC": float(roc_auc_score(y, p)),
            "Brier": float(brier_score_loss(y, p)),
            "base_rate": float(np.mean(y))}


def fit_logit(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=4000, C=1.0)
    m.fit(sc.transform(Xtr), ytr)
    return m.predict_proba(sc.transform(Xte))[:, 1]


def fit_lgbm(Xtr, ytr, Xte):
    m = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                       subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                       verbose=-1, n_jobs=4)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def run_task(d: pd.DataFrame, target: str, level: str, rows: list) -> None:
    frame = d
    if level == "day":
        # one row per day: aggregate features by taking the day's forecast shape stats,
        # which are constant within the day, plus min/max of intraday features
        agg = {c: "mean" for c in FEATURES_CAL + FEATURES_AR}
        agg.update({c: "mean" for c in FEATURES_FC})
        extra = frame.groupby("day").agg(
            resload_min=("residual_load", "min"),
            res_share_max=("res_share", "max"),
            n_low_hours=("residual_load", lambda s: int((s < s.quantile(0.25)).sum())))
        frame = frame.groupby("day").agg(agg).join(extra)
        frame[target] = d.groupby("day")[target].max()
        frame.index = pd.to_datetime(frame.index)

    y_all = frame[target]
    # daily panels have ~365 rows/year, so the minimum training size must scale
    min_train, min_test = (2000, 100) if level == "hour" else (500, 60)
    for test_year in TEST_YEARS:
        tr = frame.index.year < test_year
        te = frame.index.year == test_year
        if tr.sum() < min_train or te.sum() < min_test:
            continue
        base = float(y_all[te].mean())
        if base <= 0:
            continue

        # climatology (month x hour for hourly tasks, month for daily)
        if level == "hour":
            key_tr = [frame.index[tr].month, frame.index[tr].hour]
            key_te = pd.MultiIndex.from_arrays([frame.index[te].month, frame.index[te].hour])
        else:
            key_tr = [frame.index[tr].month]
            key_te = pd.Index(frame.index[te].month)
        clim = y_all[tr].groupby(key_tr).mean()
        p = clim.reindex(key_te).fillna(base).to_numpy()
        rows.append({"task": target, "model": "climatology", "features": "-",
                     "test_year": test_year, **score(y_all[te], p)})

        if level == "hour":
            p = frame.loc[te, "neg_h_d2"].fillna(base).to_numpy() * 0.95 + 0.02
            rows.append({"task": target, "model": "persistence(D-2)", "features": "-",
                         "test_year": test_year, **score(y_all[te], p)})

        for fname, cols in FEATURE_SETS.items():
            cols = [c for c in cols if c in frame.columns]
            X = frame[cols]
            ok_tr = tr & X.notna().all(axis=1).to_numpy()
            ok_te = te & X.notna().all(axis=1).to_numpy()
            if ok_tr.sum() < min_train or ok_te.sum() < min_test or y_all[ok_tr].sum() < 20:
                continue
            for mname, fn in (("logit", fit_logit), ("lgbm", fit_lgbm)):
                p = fn(X[ok_tr].to_numpy(), y_all[ok_tr].to_numpy(), X[ok_te].to_numpy())
                rows.append({"task": target, "model": mname, "features": fname,
                             "test_year": test_year, **score(y_all[ok_te], p)})
        print(f"  {target} {test_year} done (base {base:.3f})", flush=True)


def main() -> None:
    (PROJECT_ROOT / "outputs" / "tables").mkdir(parents=True, exist_ok=True)
    df = load_hourly()
    d, daily = build(df)
    print("panel:", d.shape, d.index.min(), "->", d.index.max())
    print("base rates: y_neg %.4f | y_run4 %.4f | y_day4(day) %.4f"
          % (d.y_neg.mean(), d.y_run4.mean(), daily.y_day4.mean()))

    rows = []
    run_task(d, "y_neg", "hour", rows)
    run_task(d, "y_run4", "hour", rows)
    run_task(d, "y_day4", "day", rows)

    res = pd.DataFrame(rows)
    res.to_csv(PROJECT_ROOT / "outputs" / "tables" / "baseline_results.csv", index=False)

    lines = ["# Baseline Results — negative-price episode forecasting\n",
             "Expanding-window rolling origin; one model per test year. Primary metric PR-AUC.",
             "No-skill PR-AUC floor = base rate (shown per task/year).\n"]
    for task, label in (("y_neg", "Task A — negative price at hour t"),
                        ("y_run4", "Task B — hour inside a negative run >= 4 h"),
                        ("y_day4", "Task C — day contains a negative run >= 4 h")):
        sub = res[res.task == task]
        if sub.empty:
            continue
        lines.append(f"\n## {label}\n")
        piv = sub.assign(m=sub.model + " [" + sub.features + "]").pivot_table(
            index="m", columns="test_year", values="PR_AUC")
        base = sub.groupby("test_year")["base_rate"].first()
        lines.append("base rate: " + ", ".join(f"{y}: {b:.3f}" for y, b in base.items()) + "\n")
        lines.append(piv.round(3).to_markdown())
        best_ar = sub[(sub.features == "cal+AR")].groupby("test_year")["PR_AUC"].max()
        best_fc = sub[(sub.features.isin(["cal+AR+FC", "cal+FC"]))].groupby("test_year")["PR_AUC"].max()
        gain = (100 * (best_fc - best_ar) / best_ar).round(1)
        lines.append("\n**Forecast-feature gain over best AR model (%):** " +
                     ", ".join(f"{y}: {g:+.1f}%" for y, g in gain.items()))

    text = "\n".join(lines)
    (PROJECT_ROOT / "experiments" / "03_baselines" / "BASELINE_NOTES.md").write_text(
        text, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(text)


if __name__ == "__main__":
    sys.exit(main())
