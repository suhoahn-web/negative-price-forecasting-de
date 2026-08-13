"""How much of the result rests on the renewable forecast, whose pre-gate publication we
cannot yet document from a primary source?

THE ISSUE, from the Regulation itself
  Article 6(1)(b) of Commission Regulation (EU) No 543/2013 requires the day-ahead TOTAL LOAD
  forecast to be published "no later than two hours before the gate closure of the day-ahead
  market in the bidding zone". Its presence in the day-ahead information set is guaranteed.

  Article 14(2)(d) sets the deadline for the WIND AND SOLAR forecast at "no later than 5 p.m.,
  one day before actual delivery takes place". That is AFTER the 12:00 CET gate closure. In
  practice German TSOs publish it in the morning of D-1, and our leakage audit (experiment 07)
  shows the series is a genuine ex-ante vintage rather than a later revision. But the
  Regulation alone does not establish the publication time, so a referee may press on it.

WHAT THIS RUNS
  A restricted feature set in which the only day-ahead forecast is the LOAD forecast, plus a
  set that keeps the renewable forecast, both on the corrected D-1 autoregressive base and
  under the monthly recalibration adopted in Section 5.4. The gap between them is exactly our
  exposure to the open question.

READING, FIXED BEFORE THE RUN
  - If the load-only set still beats the autoregressive benchmark in most cells, the central
    claim survives on legally guaranteed data alone and the renewable forecast becomes an
    enhancement rather than a load-bearing assumption.
  - If it does not, the paper depends on the publication time and must say so plainly, and
    obtaining timestamped ENTSO-E document versions moves from desirable to necessary.

Outputs
  outputs/tables/load_only.csv
  experiments/13_load_only/FINDINGS.md
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
                          FEATURES_FC_LOAD_ONLY, build, load_hourly)

TEST_YEARS = [2021, 2022, 2023, 2024, 2025]
SEED = 42
SETS = {
    "cal+AR(D-1)": FEATURES_CAL + FEATURES_AR_D1,
    "cal+AR(D-1)+load only": FEATURES_CAL + FEATURES_AR_D1 + FEATURES_FC_LOAD_ONLY,
    "cal+AR(D-1)+FC": FEATURES_CAL + FEATURES_AR_D1 + FEATURES_FC,
}


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


def prepare(d, target, level, cols_all):
    if level == "hour":
        return d
    agg = {c: "mean" for c in cols_all if c in d.columns}
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
    cols_all = sorted({c for v in SETS.values() for c in v})
    rows = []
    t0 = time.perf_counter()
    for target, level in (("y_neg", "hour"), ("y_run4", "hour"), ("y_day4", "day")):
        frame = prepare(d, target, level, cols_all)
        y = frame[target]
        min_train = 2000 if level == "hour" else 500
        for sname, cols in SETS.items():
            cols = [c for c in cols if c in frame.columns]
            X = frame[cols]
            complete = X.notna().all(axis=1).to_numpy()
            for ty in TEST_YEARS:
                te = (frame.index.year == ty) & complete
                if te.sum() < (100 if level == "hour" else 60):
                    continue
                for mname, fn in MODELS.items():
                    pred = np.full(int(te.sum()), np.nan)
                    te_idx = frame.index[te]
                    for mo in range(1, 13):
                        blk = te_idx.month == mo
                        if not blk.any():
                            continue
                        cut = pd.Timestamp(year=ty, month=mo, day=1, tz=frame.index.tz)
                        tr = np.asarray(frame.index < cut) & complete
                        if tr.sum() < min_train or y[tr].sum() < 20:
                            continue
                        sub = te.copy()
                        sub[te] = blk
                        pred[blk] = fn(X[tr].to_numpy(), y[tr].to_numpy(), X[sub].to_numpy())
                    k = np.isfinite(pred)
                    if k.sum() < 50:
                        continue
                    rows.append({"task": target, "features": sname, "model": mname,
                                 "test_year": ty, "n_features": len(cols),
                                 "PR_AUC": float(average_precision_score(
                                     y[te].to_numpy()[k], pred[k]))})
                print(f"  {target} {sname:24s} {ty} done ({time.perf_counter()-t0:.0f}s)",
                      flush=True)

    res = pd.DataFrame(rows)
    out = PROJECT_ROOT / "outputs" / "tables"
    res.to_csv(out / "load_only.csv", index=False)

    best = res.groupby(["task", "test_year", "features"])["PR_AUC"].max().unstack("features")
    best["load-only gain over AR"] = 100 * (best["cal+AR(D-1)+load only"]
                                            - best["cal+AR(D-1)"]) / best["cal+AR(D-1)"]
    best["RES adds further"] = 100 * (best["cal+AR(D-1)+FC"]
                                      - best["cal+AR(D-1)+load only"]) / best["cal+AR(D-1)+load only"]

    lines = ["# How much rests on the renewable forecast?\n",
             "Article 6(1)(b) of Regulation (EU) 543/2013 guarantees the day-ahead LOAD forecast",
             "at least two hours before gate closure. Article 14(2)(d) sets the wind and solar",
             "deadline at 17:00 on D-1, after the gate. This run bounds our exposure.\n",
             best.round(3).to_markdown(), "",
             f"\n**Load forecast alone beats the autoregressive benchmark in "
             f"{int((best['load-only gain over AR'] > 0).sum())} of {len(best)} cells**, "
             f"median {best['load-only gain over AR'].median():+.1f}%.",
             f"\n**Adding the renewable forecast improves further in "
             f"{int((best['RES adds further'] > 0).sum())} of {len(best)} cells**, "
             f"median {best['RES adds further'].median():+.1f}%."]
    text = "\n".join(lines)
    (PROJECT_ROOT / "experiments" / "13_load_only" / "FINDINGS.md").write_text(
        text, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(text)


if __name__ == "__main__":
    sys.exit(main())
