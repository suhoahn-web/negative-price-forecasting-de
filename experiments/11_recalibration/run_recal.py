"""Does recalibration frequency explain why LEAR beats our classifiers in 2024-2025?

THE FINDING THAT MOTIVATES THIS
The LEAR benchmark (experiment 10) beats our best classifier in 6 of 15 task-year cells, and
all six losses fall in 2024 and 2025 — the two most recent years, in which the base rate is
highest and rising fastest. We win 2021-2023 decisively and lose 2024-2025 consistently. A
pattern that clean is a mechanism, not noise.

THE HYPOTHESIS, FIXED BEFORE THE RUN
LEAR is recalibrated EVERY DAY, following Lago et al. (2021). Our classifiers are fitted ONCE
PER TEST YEAR on an expanding window. In a market whose negative-price incidence rose from
3.4% (2023) to 5.2% (2024) to 6.6% (2025), a model refitted daily tracks the drift within the
year while an annually fitted model cannot. If recalibration frequency is the explanation,
refitting our classifiers monthly should close most of the 2024-2025 gap and should do
comparatively little for 2021-2023.

PRE-REGISTERED READING
  - Gap closes in 2024-2025 and not in 2021-2023  -> recalibration frequency is the cause.
    We adopt monthly (or daily) recalibration and say so.
  - Gap closes everywhere                          -> we were simply under-fitting; adopt it,
    and re-examine every earlier comparison.
  - Gap does not close                             -> the cause is LEAR's richer price-lag
    structure (96 lags against our 24 AR features), not recalibration. Report LEAR as the
    stronger model on those years and do not claim otherwise.

Outputs
  outputs/tables/recalibration.csv
  experiments/11_recalibration/FINDINGS.md
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.features import (FEATURES_AR_D1, FEATURES_CAL, FEATURES_FC,  # noqa: E402
                          build, load_hourly)

TEST_YEARS = [2021, 2022, 2023, 2024, 2025]
SEED = 42
FEATS = FEATURES_CAL + FEATURES_AR_D1 + FEATURES_FC


def fit_logit(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=4000, C=1.0)
    m.fit(sc.transform(Xtr), ytr)
    return m.predict_proba(sc.transform(Xte))[:, 1]


def fit_lgbm(Xtr, ytr, Xte):
    m = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                       subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                       verbose=-1, n_jobs=6)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


MODELS = {"logit": fit_logit, "lgbm": fit_lgbm}


def evaluate(frame, target, level, freq: str) -> list:
    """freq: 'year' = one fit per test year (current design); 'month' = refit each month."""
    y_all = frame[target]
    cols = [c for c in FEATS if c in frame.columns]
    X = frame[cols]
    complete = X.notna().all(axis=1).to_numpy()
    min_train = 2000 if level == "hour" else 500
    rows = []
    for ty in TEST_YEARS:
        te_year = (frame.index.year == ty) & complete
        if te_year.sum() < (100 if level == "hour" else 60):
            continue
        for mname, fn in MODELS.items():
            pred = pd.Series(np.nan, index=frame.index)
            t0 = time.perf_counter()
            if freq == "year":
                tr = (frame.index.year < ty) & complete
                if tr.sum() < min_train:
                    continue
                pred[te_year] = fn(X[tr].to_numpy(), y_all[tr].to_numpy(),
                                   X[te_year].to_numpy())
            else:
                for mo in range(1, 13):
                    blk = te_year & (frame.index.month == mo)
                    if blk.sum() == 0:
                        continue
                    cutoff = pd.Timestamp(year=ty, month=mo, day=1, tz=frame.index.tz)
                    tr = (frame.index < cutoff) & complete
                    if tr.sum() < min_train or y_all[tr].sum() < 20:
                        continue
                    pred[blk] = fn(X[tr].to_numpy(), y_all[tr].to_numpy(),
                                   X[blk].to_numpy())
            fit_s = time.perf_counter() - t0
            ok = te_year & pred.notna().to_numpy()
            if ok.sum() < 50 or y_all[ok].sum() < 5:
                continue
            rows.append({"task": target, "test_year": ty, "model": mname, "recal": freq,
                         "PR_AUC": float(average_precision_score(y_all[ok], pred[ok])),
                         "n": int(ok.sum()), "seconds": round(fit_s, 1)})
            print(f"  {target} {ty} {mname:6s} {freq:5s} "
                  f"PR-AUC {rows[-1]['PR_AUC']:.3f} ({fit_s:.0f}s)", flush=True)
    return rows


def prepare(d, target, level):
    if level == "hour":
        return d
    agg = {c: "mean" for c in FEATS if c in d.columns}
    extra = d.groupby("day").agg(
        resload_min=("residual_load", "min"),
        res_share_max=("res_share", "max"),
        n_low_hours=("residual_load", lambda s: int((s < s.quantile(0.25)).sum())))
    frame = d.groupby("day").agg(agg).join(extra)
    frame[target] = d.groupby("day")[target].max()
    frame.index = pd.to_datetime(frame.index)
    return frame


def main() -> None:
    d, _ = build(load_hourly())
    rows = []
    for target, level in (("y_neg", "hour"), ("y_run4", "hour"), ("y_day4", "day")):
        frame = prepare(d, target, level)
        for freq in ("year", "month"):
            rows += evaluate(frame, target, level, freq)

    res = pd.DataFrame(rows)
    out = PROJECT_ROOT / "outputs" / "tables"
    res.to_csv(out / "recalibration.csv", index=False)

    best = res.groupby(["task", "test_year", "recal"])["PR_AUC"].max().unstack("recal")
    best["delta"] = best["month"] - best["year"]
    lear = pd.read_csv(out / "lear_comparison.csv")
    lear_best = lear.groupby(["task", "test_year"])["PR_AUC"].max()
    best["LEAR"] = [lear_best.get(i, np.nan) for i in best.index]
    best["beats_LEAR_year"] = best["year"] > best["LEAR"]
    best["beats_LEAR_month"] = best["month"] > best["LEAR"]

    yrs = best.index.get_level_values("test_year")
    lines = ["# Recalibration frequency — does it explain LEAR's advantage?\n",
             "Pre-registered reading is in the module docstring.\n",
             best.round(3).to_markdown(),
             "",
             f"\n**Cells beaten by LEAR under annual refitting:** "
             f"{int((~best.beats_LEAR_year).sum())} of {len(best)}",
             f"\n**Cells beaten by LEAR under monthly refitting:** "
             f"{int((~best.beats_LEAR_month).sum())} of {len(best)}",
             f"\n**Mean gain from monthly refitting, 2021-2023:** "
             f"{best.loc[yrs <= 2023, 'delta'].mean():+.3f}",
             f"\n**Mean gain from monthly refitting, 2024-2025:** "
             f"{best.loc[yrs >= 2024, 'delta'].mean():+.3f}"]
    text = "\n".join(lines)
    (PROJECT_ROOT / "experiments" / "11_recalibration" / "FINDINGS.md").write_text(
        text, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(text)


if __name__ == "__main__":
    sys.exit(main())
