"""Score the LEAR point forecasts as a negative-price classifier and compare with our models.

Separated from `run_lear_benchmark.py` so that scoring can be re-run and revised without
repeating the 1,826 daily recalibrations.

Two scoring routes, both strictly backward-looking (see §5.3 of the manuscript):

  LEAR-rank  score = -p_hat. A pure ranking, which is all PR-AUC requires, with no
             distributional assumption. This is the STRONGEST honest reading of LEAR for a
             rank-based metric and is the one we report as the benchmark.
  LEAR-prob  P(price < 0) = F_hat(-p_hat), with F_hat the empirical cdf of LEAR's own
             forecast errors over the preceding 180 forecast days, pooled within six-hour
             blocks of the day and expanding from a 30-day minimum. Only errors realised
             before the scored day enter the pool.

Derived targets. LEAR produces an hourly negative-price score; the run-length and daily
targets are built from it under assumptions chosen to be GENEROUS TO LEAR:
  y_run4  for each of the four 4-hour windows containing hour t, take the weakest hour's
          score; the hour's score is the best of those four windows. Taking the minimum
          rather than the product of four probabilities is an upper bound on P(all four
          negative), so it cannot understate LEAR. The product variant is computed as a
          robustness check and the better of the two is reported.
  y_day4  the maximum of the y_run4 score over the 24 hours of the day.

Outputs
  outputs/tables/lear_comparison.csv
  experiments/10_lear/FINDINGS.md
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.features import build, load_hourly  # noqa: E402
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "08_infoset_dm"))
from run_infoset_dm import dm_hln  # noqa: E402

TEST_YEARS = [2021, 2022, 2023, 2024, 2025]
RESID_MAX, RESID_MIN = 180, 30
EPS = 1e-6


def empirical_prob(pred: pd.Series, actual: pd.Series) -> pd.Series:
    """P(price<0) from LEAR's own past forecast errors, pooled in 6-hour blocks.

    Rows within a block are time-ordered, so the backward pool is a contiguous slice and the
    whole loop is O(n) rather than O(n^2).
    """
    err = actual.reindex(pred.index) - pred
    days = pred.index.normalize()
    uniq = pd.DatetimeIndex(sorted(days.unique()))
    pos_of = pd.Series(np.arange(len(uniq)), index=uniq)
    out = pd.Series(np.nan, index=pred.index)

    block = pred.index.hour // 6
    for b in range(4):
        m = block == b
        idx = pred.index[m]
        e = err[m].to_numpy(float)
        ph = pred[m].to_numpy(float)
        dp = pos_of.reindex(days[m]).to_numpy()          # day position, non-decreasing
        lo = np.searchsorted(dp, dp - RESID_MAX, side="left")
        hi = np.searchsorted(dp, dp, side="left")        # strictly before the scored day
        vals = np.full(len(idx), np.nan)
        for i in range(len(idx)):
            pool = e[lo[i]:hi[i]]
            pool = pool[np.isfinite(pool)]
            if (dp[i] - dp[lo[i]]) < RESID_MIN or pool.size < 20 or not np.isfinite(ph[i]):
                continue
            vals[i] = float(np.mean(pool < -ph[i]))
        out.loc[idx] = vals
    return out


def run4_score(p_hourly: pd.Series, mode: str = "min") -> pd.Series:
    """Best over the four 4-hour windows containing hour t."""
    v = p_hourly.to_numpy(float)
    n = len(v)
    best = np.full(n, np.nan)
    for start in range(-3, 1):                      # window [t+start, t+start+3]
        w = np.full(n, np.nan)
        sl = [np.roll(v, -(start + k)) for k in range(4)]
        M = np.vstack(sl)
        with np.errstate(invalid="ignore"):
            allnan = np.all(~np.isfinite(M), axis=0)
            w = np.where(allnan, np.nan,
                         np.nanmin(np.where(np.isfinite(M), M, np.inf), axis=0) if mode == "min"
                         else np.nanprod(np.where(np.isfinite(M), M, 1.0), axis=0))
        edge = max(0, -(start)), max(0, start + 3)
        if edge[0]:
            w[:edge[0]] = np.nan
        if edge[1]:
            w[-edge[1]:] = np.nan
        best = np.fmax(best, w)
    return pd.Series(best, index=p_hourly.index)


def score(y, p) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    # a handful of timestamps exist in the LEAR forecast grid but not in the price panel
    # (clock changes); drop them from both sides rather than imputing
    ok = np.isfinite(p) & np.isfinite(y)
    y, p = y[ok], p[ok]
    if ok.sum() == 0:
        return {"PR_AUC": np.nan, "ROC_AUC": np.nan, "n": 0}
    if len(np.unique(y)) < 2:
        return {"PR_AUC": np.nan, "ROC_AUC": np.nan, "n": int(ok.sum())}
    return {"PR_AUC": float(average_precision_score(y, p)),
            "ROC_AUC": float(roc_auc_score(y, p)),
            "n": int(ok.sum()), "base_rate": float(np.mean(y))}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="", help="'' for the full model, '_noexog' for the "
                                                 "price-history-only variant")
    args = ap.parse_args()
    tag = "LEAR-noexog" if args.suffix else "LEAR"

    out_t = PROJECT_ROOT / "outputs" / "tables"
    out_p = PROJECT_ROOT / "outputs" / "preds"
    point = pd.read_parquet(out_p / f"lear_point{args.suffix}.parquet")
    d, _ = build(load_hourly())
    d.index = d.index.tz_localize(None)
    actual = d["price"]

    pred = point["lear_ens"]
    mae = float((actual.reindex(pred.index) - pred).abs().mean())
    print(f"LEAR ensemble MAE {mae:.2f} EUR/MWh over {len(pred)} hours", flush=True)

    prob = empirical_prob(pred, actual)
    print(f"probability coverage {prob.notna().mean():.3f}", flush=True)

    scores = {f"{tag}-rank": -pred, f"{tag}-prob": prob}
    rows = []
    for name, s_hour in scores.items():
        s_run = {m: run4_score(s_hour, m) for m in ("min", "prod")}
        for ty in TEST_YEARS:
            te = s_hour.index.year == ty
            idx = s_hour.index[te]
            y_neg = d["y_neg"].reindex(idx).to_numpy()
            y_run = d["y_run4"].reindex(idx).to_numpy()
            rows.append({"model": name, "task": "y_neg", "test_year": ty,
                         **score(y_neg, s_hour[te].to_numpy())})
            best = max((score(y_run, s_run[m][te].to_numpy()) for m in ("min", "prod")),
                       key=lambda r: (r["PR_AUC"] if np.isfinite(r["PR_AUC"]) else -1))
            rows.append({"model": name, "task": "y_run4", "test_year": ty, **best})
            # daily: max of the run score over the day
            dser = pd.Series(s_run["min"][te].to_numpy(), index=idx)
            day_s = dser.groupby(dser.index.normalize()).max()
            day_y = d["y_day4"].reindex(idx)
            day_y = day_y.groupby(day_y.index.normalize()).max()
            j = day_s.index.intersection(day_y.index)
            rows.append({"model": name, "task": "y_day4", "test_year": ty,
                         **score(day_y.loc[j].to_numpy(), day_s.loc[j].to_numpy())})

    lear = pd.DataFrame(rows)
    lear["mae"] = mae
    lear.to_csv(out_t / f"lear_comparison{args.suffix}.csv", index=False)

    ours = pd.read_csv(out_t / "infoset_comparison.csv")
    best_ours = (ours[ours.features == "cal+AR(D-1)+FC"]
                 .groupby(["task", "test_year"])["PR_AUC"].max())
    best_ar = (ours[ours.features == "cal+AR(D-1)"]
               .groupby(["task", "test_year"])["PR_AUC"].max())

    lines = [f"# {tag} Benchmark\n",
             f"Authors' own `epftoolbox` implementation. Ensemble of calibration windows "
             f"{{56, 84, 364, 728}} days, recalibrated daily, {len(pred):,} hourly forecasts.",
             f"Ensemble MAE on the price level: **{mae:.2f} EUR/MWh**.\n",
             "\n## PR-AUC comparison\n",
             f"| task | year | base | {tag}-rank | {tag}-prob | ours AR(D-1) | ours full |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for task in ("y_neg", "y_run4", "y_day4"):
        for ty in TEST_YEARS:
            r = lear[(lear.task == task) & (lear.test_year == ty)]
            gv = lambda m: (r[r.model == m]["PR_AUC"].iloc[0]  # noqa: E731
                            if len(r[r.model == m]) else np.nan)
            br = r["base_rate"].dropna()
            lines.append(
                f"| {task} | {ty} | {br.iloc[0]:.3f} | {gv(f'{tag}-rank'):.3f} | "
                f"{gv(f'{tag}-prob'):.3f} | {best_ar.get((task, ty), np.nan):.3f} | "
                f"**{best_ours.get((task, ty), np.nan):.3f}** |")

    # if both variants exist, contrast them directly
    other = out_t / ("lear_comparison.csv" if args.suffix else "lear_comparison_noexog.csv")
    if other.exists():
        o = pd.read_csv(other)
        a = lear.groupby(["task", "test_year"])["PR_AUC"].max()
        b = o.groupby(["task", "test_year"])["PR_AUC"].max()
        withx, without = (b, a) if args.suffix else (a, b)
        lines += ["\n\n## The day-ahead forecasts inside LEAR itself\n",
                  "Same model, same calibration windows, same daily recalibration; the only",
                  "difference is whether the published day-ahead renewable and load forecasts",
                  "are supplied as exogenous inputs. This is the cleanest available test of the",
                  "information-set claim, because the model is held fixed.\n",
                  "| task | year | LEAR with forecasts | LEAR price-only | gain |",
                  "|---|---:|---:|---:|---:|"]
        for k in sorted(set(withx.index) & set(without.index)):
            w, wo = withx[k], without[k]
            lines.append(f"| {k[0]} | {k[1]} | {w:.3f} | {wo:.3f} | "
                         f"{100*(w-wo)/wo:+.1f}% |")
        gains = [(withx[k] - without[k]) / without[k] for k in
                 sorted(set(withx.index) & set(without.index))]
        lines.append(f"\n**Positive in {sum(g > 0 for g in gains)} of {len(gains)} cells; "
                     f"median gain {100*float(np.median(gains)):+.1f}%.**")

    text = "\n".join(lines)
    (PROJECT_ROOT / "experiments" / "10_lear" /
     f"FINDINGS{args.suffix}.md").write_text(text, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(text)


if __name__ == "__main__":
    sys.exit(main())
