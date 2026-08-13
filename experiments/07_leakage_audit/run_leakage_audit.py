"""Leakage audit for the negative-price prediction result.

The headline claim (day-ahead forecasts beat autoregressive baselines by +42% to +706%
PR-AUC) is exactly the kind of large result that look-ahead bias produces. This script
tries to FALSIFY it with four independent tests.

T1 vintage separation   day-ahead vs intraday vs current forecasts must differ
                        (run separately; result: mean abs diff 503-766 MW, not identical)
T2 realistic error      day-ahead forecast vs realised generation must show genuine
                        forecast error (result: wind nRMSE 3.61%, solar 3.25% of peak)
T3 information ordering ACTUAL generation must beat the DAY-AHEAD FORECAST as a predictor.
                        If the forecast performs as well as the actual, the "forecast" is
                        carrying information it should not have.
T4 placebo shift        replacing each hour's forecast with the forecast for the SAME HOUR
                        N DAYS LATER must destroy performance. If a future-shifted forecast
                        still predicts today, the label or the join is misaligned.
T5 anti-placebo         shifting the TARGET forward by N days (predicting an unrelated
                        future day) must collapse to the base rate.

Outputs: experiments/07_leakage_audit/AUDIT_NOTES.md
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.features import FEATURES_CAL, build, load_hourly  # noqa: E402

RAW = PROJECT_ROOT / "data" / "raw"
TRAIN_END, TEST_YEAR = 2023, 2024
SEED = 42


def fetch_actuals(years) -> pd.DataFrame:
    """Realised generation and load (for T3 only — never used as a model feature)."""
    cache = RAW / "actuals_de.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    frames = []
    for y in years:
        r = requests.get("https://api.energy-charts.info/public_power",
                         params={"country": "de", "start": f"{y}-01-01", "end": f"{y}-12-31"},
                         timeout=300)
        j = r.json()
        idx = pd.to_datetime(j["unix_seconds"], unit="s", utc=True)
        d = {it["name"]: it["data"] for it in j["production_types"]}
        frames.append(pd.DataFrame(d, index=idx))
        time.sleep(3)
        print(f"  actuals {y} ok", flush=True)
    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="first")]
    out.to_parquet(cache)
    return out


def pr_auc(X: pd.DataFrame, y: pd.Series, years: pd.Index) -> float:
    tr = (years <= TRAIN_END) & X.notna().all(axis=1).to_numpy()
    te = (years == TEST_YEAR) & X.notna().all(axis=1).to_numpy()
    if tr.sum() < 2000 or te.sum() < 500 or y[tr].sum() < 30:
        return np.nan
    m = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                       subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                       verbose=-1, n_jobs=4)
    m.fit(X[tr].to_numpy(), y[tr].to_numpy())
    p = m.predict_proba(X[te].to_numpy())[:, 1]
    return float(average_precision_score(y[te], p))


def main() -> None:
    HERE = PROJECT_ROOT / "experiments" / "07_leakage_audit"
    HERE.mkdir(parents=True, exist_ok=True)
    d, _ = build(load_hourly())
    y = d["y_neg"]
    years = d.index.year
    base = float(y[years == TEST_YEAR].mean())

    lines = ["# Leakage Audit — negative-price prediction\n",
             f"Train <= {TRAIN_END}, test {TEST_YEAR}. LightGBM, PR-AUC. "
             f"No-skill floor = base rate {base:.4f}.\n",
             "## T1 vintage separation (run separately)\n",
             "day-ahead vs intraday: mean abs diff 503 MW; vs current: 766 MW; not identical."
             " -> the day-ahead label is a distinct vintage, not a relabelled latest revision.\n",
             "## T2 realistic forecast error (run separately)\n",
             "wind onshore nRMSE 3.61% of peak, solar 3.25%; corr 0.987/0.994; not identical"
             " to actuals. -> genuine forecast error of the expected magnitude.\n"]

    fc_cols = ["solar_fc", "wind_onshore_fc", "wind_offshore_fc", "load_fc",
               "res_fc", "residual_load", "res_share"]

    # ---------- T3 information ordering ----------
    print("T3: fetching actuals", flush=True)
    act = fetch_actuals(range(2019, 2026))
    act_h = act.resample("h").mean()
    a = pd.DataFrame(index=d.index)
    a["solar_act"] = act_h.get("Solar")
    a["wind_on_act"] = act_h.get("Wind onshore")
    a["wind_off_act"] = act_h.get("Wind offshore")
    a["load_act"] = act_h.get("Load")
    a["res_act"] = a[["solar_act", "wind_on_act", "wind_off_act"]].sum(axis=1, min_count=1)
    if a["load_act"].notna().any():
        a["residual_act"] = a["load_act"] - a["res_act"]
        a["res_share_act"] = a["res_act"] / a["load_act"].replace(0, np.nan)

    X_cal = d[FEATURES_CAL]
    X_fc = pd.concat([X_cal, d[fc_cols]], axis=1)
    act_cols = [c for c in a.columns if a[c].notna().sum() > 10000]
    X_act = pd.concat([X_cal, a[act_cols]], axis=1)

    s_cal = pr_auc(X_cal, y, years)
    s_fc = pr_auc(X_fc, y, years)
    s_act = pr_auc(X_act, y, years)
    lines.append("## T3 information ordering\n")
    lines.append("| feature set | PR-AUC |\n|---|---:|")
    lines.append(f"| calendar only | {s_cal:.3f} |")
    lines.append(f"| calendar + DAY-AHEAD FORECAST | {s_fc:.3f} |")
    lines.append(f"| calendar + REALISED generation (oracle) | {s_act:.3f} |")
    verdict3 = "PASS" if (s_act > s_fc + 0.01) else "FAIL"
    lines.append(f"\nRealised generation should strictly dominate the forecast. "
                 f"**{verdict3}** (oracle - forecast = {s_act - s_fc:+.3f}). "
                 f"Actual columns used: {act_cols}\n")

    # ---------- T4 placebo: forecast shifted to a later day ----------
    lines.append("## T4 placebo — forecast taken from N days LATER\n")
    lines.append("| shift | PR-AUC |\n|---|---:|")
    lines.append(f"| 0 days (real) | {s_fc:.3f} |")
    for days in (1, 3, 7):
        shifted = d[fc_cols].shift(-24 * days)
        Xp = pd.concat([X_cal, shifted], axis=1)
        lines.append(f"| +{days} days | {pr_auc(Xp, y, years):.3f} |")
    lines.append("\nA future-shifted forecast should lose most of its skill. Some skill "
                 "remains by construction because weather is autocorrelated over days.\n")

    # ---------- T5 anti-placebo: target shifted ----------
    lines.append("## T5 anti-placebo — target taken from N days LATER\n")
    lines.append("| target shift | PR-AUC |\n|---|---:|")
    lines.append(f"| 0 days (real) | {s_fc:.3f} |")
    for days in (1, 3, 7):
        y_shift = y.shift(-24 * days)
        ok = y_shift.notna()
        lines.append(f"| +{days} days | "
                     f"{pr_auc(X_fc[ok], y_shift[ok].astype(int), years[ok]):.3f} |")
    lines.append(f"\nBase rate is {base:.3f}; a shifted target should fall toward it.\n")

    text = "\n".join(lines)
    (HERE / "AUDIT_NOTES.md").write_text(text, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(text)


if __name__ == "__main__":
    sys.exit(main())
