"""LEAR benchmark — the state-of-the-art open-access model demanded by the Lago checklist.

WHY THIS EXISTS
Lago, Marcjasz, De Schutter & Weron (2021, *Applied Energy* 293:116983) require that
"any new model is tested against state-of-the-art open-access models" and that
"state-of-the-art and free toolboxes are used for modeling the benchmark models".
Our baseline suite tested only against autoregressive logit/LightGBM, so we failed both
rows. This closes them.

WHAT IS USED
The authors' OWN implementation: `epftoolbox.models._lear.LEAR`, installed from
https://github.com/jeslago/epftoolbox. Nothing is reimplemented, so the benchmark cannot
be accused of being a straw man. (The package's `models/__init__.py` imports the
TensorFlow DNN, which we do not need; the LEAR module is therefore loaded directly by
file path. No source is modified.)

HOW A POINT MODEL BECOMES A NEGATIVE-PRICE CLASSIFIER
LEAR forecasts the price LEVEL. Stream 1 of the literature review flagged the obvious
referee question: why classify directly instead of deriving P(price < 0) from a
distributional or point regression (cf. Marcjasz et al. 2023, *Energy Economics*
125:106843)? Two answers are produced here, both strictly out of sample:

  LEAR-rank  score = -p_hat. A pure ranking, which is all PR-AUC needs, and free of any
             distributional assumption.
  LEAR-prob  P(price < 0) = F_hat(-p_hat), where F_hat is the EMPIRICAL cdf of LEAR's own
             forecast errors over the preceding 180 forecast days (expanding from a
             30-day minimum). Only past errors are ever used, so there is no leakage from
             the evaluated year. Hour-of-day is respected by pooling errors within the
             same 6-hour block.

For the run-length and daily targets, the derived hourly probabilities are combined:
  y_run4  P(hour in a >=4h run) is approximated by the minimum of P(neg) over every
          4-hour window containing the hour -- the probability that ALL hours of at least
          one such window are negative, under a working independence assumption. The
          assumption is generous to LEAR, not to us.
  y_day4  max over the day of the y_run4 score.

DEVIATION FROM LAGO ET AL., STATED EXPLICITLY
They ensemble calibration windows {56, 84, 1092, 1456} days. Our price history begins
2019-01-01, so a 1456-day window could not produce a forecast before 2023 and would cost
us three of five test years. We use {56, 84, 364, 728}. This is a shorter-memory ensemble
and, if anything, understates LEAR on long-memory effects; it is reported as a limitation.

Outputs
  outputs/preds/lear_point.parquet      per-hour LEAR point forecasts, one column per window
  outputs/preds/lear_scores_<task>.parquet  derived classifier scores
  outputs/tables/lear_comparison.csv    PR-AUC etc. vs our models
  experiments/10_lear/FINDINGS.md
"""

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.features import build, load_hourly  # noqa: E402

CAL_WINDOWS = [56, 84, 364, 728]
FORECAST_START = "2021-01-01"
FORECAST_END = "2025-12-31"
RESID_MAX = 180      # days of past forecast errors used for the empirical cdf
RESID_MIN = 30       # below this the day is not scored


def load_lear_class():
    """Load epftoolbox's LEAR without triggering the TensorFlow import in models/__init__."""
    import epftoolbox
    p = os.path.join(os.path.dirname(epftoolbox.__file__), "models", "_lear.py")
    spec = importlib.util.spec_from_file_location("eft_lear", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.LEAR


def build_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    """epftoolbox format: hourly index, columns Price / Exogenous 1 / Exogenous 2.

    Exogenous 1 = total day-ahead RES forecast, Exogenous 2 = day-ahead load forecast.
    Both are published before the 12:00 CET D-1 gate, mirroring the zonal-load and zonal-
    generation forecasts used in epftoolbox's own German dataset.
    """
    d, _ = build(load_hourly())
    df = pd.DataFrame({
        "Price": d["price"].to_numpy(),
        "Exogenous 1": d["res_fc"].to_numpy(),
        "Exogenous 2": d["load_fc"].to_numpy(),
    }, index=d.index.tz_localize(None))
    df = df[~df.index.duplicated(keep="first")].asfreq("h")
    df[["Exogenous 1", "Exogenous 2"]] = df[["Exogenous 1", "Exogenous 2"]].ffill()
    df["Price"] = df["Price"].interpolate(limit=3)
    return df, d


def _assemble(results) -> pd.DataFrame:
    idx, cols = [], {cw: [] for cw in CAL_WINDOWS}
    for day, row in sorted(results, key=lambda r: r[0]):
        idx.append(pd.date_range(day, periods=24, freq="h"))
        for cw in CAL_WINDOWS:
            cols[cw].append(row[cw])
    if not idx:
        return pd.DataFrame()
    index = pd.DatetimeIndex(np.concatenate([i.to_numpy() for i in idx]))
    out = pd.DataFrame({f"lear_{cw}": np.concatenate(cols[cw]) for cw in CAL_WINDOWS},
                       index=index)
    out["lear_ens"] = out[[f"lear_{cw}" for cw in CAL_WINDOWS]].mean(axis=1)
    return out


def forecast_all(df: pd.DataFrame, days: pd.DatetimeIndex, n_jobs: int,
                 ckpt: Path | None = None) -> pd.DataFrame:
    """Fit LEAR for every day. Checkpoints after each chunk so a stopped run resumes."""
    LEAR = load_lear_class()

    done = pd.DataFrame()
    if ckpt is not None and ckpt.exists():
        done = pd.read_parquet(ckpt)
        have = set(pd.DatetimeIndex(done.index).normalize().unique())
        days = pd.DatetimeIndex([d for d in days if d not in have])
        print(f"  resuming: {len(have)} days already on disk, {len(days)} to go", flush=True)

    def one_day(day):
        row = {}
        for cw in CAL_WINDOWS:
            try:
                yp = LEAR(calibration_window=cw).recalibrate_and_forecast_next_day(
                    df=df, calibration_window=cw, next_day_date=day)
                row[cw] = np.asarray(yp, float).ravel()
            except Exception:
                row[cw] = np.full(24, np.nan)
        return day, row

    t0 = time.perf_counter()
    results = []
    if n_jobs > 1:
        from joblib import Parallel, delayed
        chunk = 50
        for i in range(0, len(days), chunk):
            part = days[i:i + chunk]
            results += Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(one_day)(dd) for dd in part)
            n = len(results)
            el = time.perf_counter() - t0
            print(f"  {n}/{len(days)} days | {el/n:.2f} s/day | "
                  f"ETA {(len(days)-n)*el/n/60:.1f} min", flush=True)
            if ckpt is not None:
                part = _assemble(results)
                if not done.empty:
                    part = pd.concat([done, part]).sort_index()
                part.to_parquet(ckpt)
    else:
        for k, dd in enumerate(days, 1):
            results.append(one_day(dd))
            if k % 20 == 0:
                el = time.perf_counter() - t0
                print(f"  {k}/{len(days)} days | {el/k:.2f} s/day | "
                      f"ETA {(len(days)-k)*el/k/60:.1f} min", flush=True)

    out = _assemble(results)
    if ckpt is not None and ckpt.exists():
        prev = pd.read_parquet(ckpt)
        out = pd.concat([prev, out]).sort_index()
        out = out[~out.index.duplicated(keep="last")]
    return out


def derive_probabilities(point: pd.DataFrame, actual: pd.Series) -> pd.Series:
    """P(price<0) from LEAR's own PAST forecast errors. Strictly backward-looking."""
    pred = point["lear_ens"]
    err = (actual.reindex(pred.index) - pred)            # actual - forecast
    day = pred.index.normalize()
    block = pred.index.hour // 6
    days = day.unique().sort_values()
    day_pos = {d: i for i, d in enumerate(days)}

    prob = pd.Series(np.nan, index=pred.index)
    # pre-group errors by (day, block) so the rolling pool is cheap
    err_df = pd.DataFrame({"day": day, "block": block, "err": err.to_numpy()},
                          index=pred.index)
    for b in range(4):
        sub = err_df[err_df.block == b]
        sub_days = sub["day"].to_numpy()
        e = sub["err"].to_numpy()
        p_hat = pred.loc[sub.index].to_numpy()
        out = np.full(len(sub), np.nan)
        for i in range(len(sub)):
            pos = day_pos[sub_days[i]]
            lo = max(0, pos - RESID_MAX)
            mask = (np.array([day_pos[dd] for dd in sub_days]) >= lo) & \
                   (np.array([day_pos[dd] for dd in sub_days]) < pos)
            pool = e[mask]
            pool = pool[np.isfinite(pool)]
            n_days = len(np.unique(sub_days[mask]))
            if n_days < RESID_MIN or pool.size < 20:
                continue
            # P(actual < 0) = P(err < -p_hat)
            out[i] = float(np.mean(pool < -p_hat[i]))
        prob.loc[sub.index] = out
    return prob


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=6)
    ap.add_argument("--limit-days", type=int, default=0, help="timing test only")
    ap.add_argument("--forecast-only", action="store_true",
                    help="fit and save point forecasts, leave scoring to score_lear.py")
    ap.add_argument("--no-exog", action="store_true",
                    help="drop the day-ahead forecast inputs, leaving LEAR on price history "
                         "alone. This is the STRONGEST possible version of the information "
                         "set used by Loizidis et al. (2025), and isolates how much of LEAR's "
                         "skill comes from the published forecasts rather than from its "
                         "richer price-lag structure.")
    ap.add_argument("--out-suffix", type=str, default="",
                    help="suffix for the output parquet, e.g. '_noexog'")
    args = ap.parse_args()

    out_p = PROJECT_ROOT / "outputs" / "preds"
    out_t = PROJECT_ROOT / "outputs" / "tables"
    out_p.mkdir(parents=True, exist_ok=True)
    out_t.mkdir(parents=True, exist_ok=True)

    df, panel = build_frame()
    if args.no_exog:
        df = df[["Price"]].copy()
        print("NO-EXOG mode: LEAR sees price history only", flush=True)
    print("frame:", df.shape, df.index.min(), "->", df.index.max(), flush=True)
    print("nan Price:", int(df.Price.isna().sum()),
          "| exogenous columns:", [c for c in df.columns if c != "Price"], flush=True)

    days = pd.date_range(FORECAST_START, FORECAST_END, freq="D")
    if args.limit_days:
        days = days[:args.limit_days]
    print(f"forecasting {len(days)} days x {len(CAL_WINDOWS)} windows "
          f"x 24 hourly LASSO fits", flush=True)

    ckpt = None if args.limit_days else out_p / f"lear_ckpt{args.out_suffix}.parquet"
    point = forecast_all(df, days, args.n_jobs, ckpt)
    point.to_parquet(out_p / f"lear_point{args.out_suffix}.parquet")
    print("saved point forecasts:", point.shape, flush=True)

    actual = df["Price"]
    mae = (actual.reindex(point.index) - point["lear_ens"]).abs().mean()
    print(f"LEAR ensemble MAE over the forecast span: {mae:.2f} EUR/MWh", flush=True)
    if args.limit_days or args.forecast_only:
        print("point forecasts done -- scoring is handled by score_lear.py")
        return

    prob = derive_probabilities(point, actual)
    scores = pd.DataFrame({"lear_rank": -point["lear_ens"], "lear_prob": prob})
    scores.to_parquet(out_p / "lear_scores_hourly.parquet")
    print("derived probabilities; coverage %.3f" % prob.notna().mean(), flush=True)


if __name__ == "__main__":
    sys.exit(main())
