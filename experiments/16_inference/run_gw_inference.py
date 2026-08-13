"""Forecast-comparison inference: Giacomini-White, block bootstrap, and FDR control.

WHY THIS REPLACES THE EARLIER TESTS
The manuscript reported Diebold-Mariano with the Harvey-Leybourne-Newbold correction on daily
Brier differentials. Three objections are fair and are addressed here.

  1. The models are re-estimated on a rolling schedule (monthly), so the object being compared
     is a forecasting METHOD including its estimation scheme, not a fixed model. Giacomini &
     White (2006) is the test designed for exactly that case; DM assumes the forecasts are
     given. GW is therefore the primary test here and DM is demoted to a robustness check.
  2. The HLN correction was derived for a rectangular-window variance estimator truncated at
     lag h-1, not for the Newey-West HAC we pair it with. A stationary block bootstrap makes no
     such assumption and is reported alongside.
  3. A tally of how many of fifty tests are individually significant is not evidence about the
     family. We control the false discovery rate across the family with Benjamini-Hochberg and
     report the count that survives.

WHAT IS TESTED
For each target and test year, the daily Brier loss differential between the autoregressive
specification and the same specification plus the published day-ahead forecasts. The
differential is loss(AR) minus loss(AR + forecasts), so a POSITIVE mean means the forecast
covariates lower the loss and therefore favours them.

  GW unconditional : test of E[d_t] = 0 with a HAC variance (Newey-West, Bartlett).
  GW conditional   : regress d_t on instruments [1, d_{t-1}] and test both coefficients jointly.
                     This asks whether the difference is predictable, which is the conditional
                     predictive ability hypothesis.
  block bootstrap  : stationary bootstrap of the daily differential, 5000 replications, mean
                     block length 5 and 10 and 20 days for sensitivity.

Outputs
  outputs/tables/gw_inference.csv
  experiments/16_inference/FINDINGS.md
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEED = 42
EPS = 1e-6
TASKS = ["y_neg", "y_run3", "y_run4", "y_run6", "y_day4"]
YEARS = [2021, 2022, 2023, 2024, 2025]

# Two comparison families, each answering a different question and each tested on the same
# 5 targets x 5 years grid so the denominators in the manuscript are identical. The FDR
# correction is applied WITHIN a family: pooling them would control the wrong error rate,
# because a discovery about the information set is not a discovery about the forecast
# covariates.
FAMILIES = {
    # does moving from a D-2 to a D-1 information set improve the autoregressive benchmark?
    "infoset": ("lgbm|cal+AR(D-2)", "lgbm|cal+AR(D-1)"),
    # given the correct information set, do the published day-ahead forecasts add to it?
    "forecast": ("lgbm|cal+AR(D-1)", "lgbm|cal+AR(D-1)+FC"),
}


def hac_var(x: np.ndarray, lags: int | None = None) -> float:
    x = x - x.mean()
    T = len(x)
    if lags is None:
        lags = int(np.floor(4 * (T / 100.0) ** (2 / 9)))
    v = float(x @ x) / T
    for k in range(1, lags + 1):
        g = float(x[k:] @ x[:-k]) / T
        v += 2 * (1 - k / (lags + 1.0)) * g
    return max(v, 1e-18)


def gw_unconditional(d: np.ndarray) -> tuple:
    T = len(d)
    s = np.sqrt(hac_var(d) / T)
    z = d.mean() / s
    return float(z), float(2 * (1 - stats.norm.cdf(abs(z))))


def gw_conditional(d: np.ndarray) -> tuple:
    """Wald test that E[d_t | 1, d_{t-1}] = 0, with a HAC covariance."""
    y = d[1:]
    X = np.column_stack([np.ones(len(y)), d[:-1]])
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    T = len(y)
    lags = int(np.floor(4 * (T / 100.0) ** (2 / 9)))
    S = (X * e[:, None]).T @ (X * e[:, None]) / T
    for k in range(1, lags + 1):
        u = (X[k:] * e[k:, None]).T @ (X[:-k] * e[:-k, None]) / T
        S += (1 - k / (lags + 1.0)) * (u + u.T)
    V = T * XtX_inv @ S @ XtX_inv
    W = float(b @ np.linalg.pinv(V) @ b)
    return W, float(stats.chi2.sf(W, df=2))


def stationary_bootstrap(d: np.ndarray, mean_block: int, reps: int, rng) -> np.ndarray:
    T = len(d)
    p = 1.0 / mean_block
    out = np.empty(reps)
    for r in range(reps):
        idx = np.empty(T, dtype=int)
        i = rng.integers(T)
        for t in range(T):
            idx[t] = i
            if rng.random() < p:
                i = rng.integers(T)
            else:
                i = (i + 1) % T
        out[r] = d[idx].mean()
    return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    # Giacomini & White (2006) require the maximum estimation sample size to remain finite as
    # the out-of-sample count grows (their Comment 2). The monthly-refit predictions in
    # preds/monthly use an EXPANDING window, which does not satisfy that condition, so the GW
    # test is only licensed on the rolling-window predictions. Default accordingly, and keep
    # the expanding-window run reachable so the two schemes can be compared.
    ap.add_argument("--preds", default="rolling730",
                    help="subdirectory of outputs/preds to test (default: rolling730, the "
                         "fixed-length window the GW asymptotics require)")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    pred_dir = PROJECT_ROOT / "outputs" / "preds" / args.preds
    suffix = "" if args.preds == "rolling730" else f"_{args.preds}"
    print(f"testing predictions in {pred_dir}", flush=True)
    rows = []

    for fam, (A_COL, B_COL) in FAMILIES.items():
      for task in TASKS:
        for year in YEARS:
            f = pred_dir / f"{task}_{year}.parquet"
            if not f.exists():
                continue
            P = pd.read_parquet(f)
            if A_COL not in P.columns or B_COL not in P.columns:
                continue
            m = P[A_COL].notna() & P[B_COL].notna()
            y = P.loc[m, "y"].to_numpy(float)
            a = np.clip(P.loc[m, A_COL].to_numpy(float), EPS, 1 - EPS)
            b = np.clip(P.loc[m, B_COL].to_numpy(float), EPS, 1 - EPS)
            la, lb = (a - y) ** 2, (b - y) ** 2
            s = pd.Series(la - lb, index=P.index[m])
            daily = (s.groupby(s.index.normalize()).mean()
                     if task != "y_day4" else s).to_numpy()
            daily = daily[np.isfinite(daily)]
            if len(daily) < 60:
                continue

            zu, pu = gw_unconditional(daily)
            wc, pc = gw_conditional(daily)
            boot = {}
            for mb in (5, 10, 20):
                bs = stationary_bootstrap(daily, mb, 5000, rng)
                boot[mb] = (float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975)))
            rows.append({
                "family": fam, "task": task, "test_year": year, "n_days": len(daily),
                "mean_diff": float(daily.mean()),
                "gw_uncond_z": zu, "gw_uncond_p": pu,
                "gw_cond_W": wc, "gw_cond_p": pc,
                "boot5_lo": boot[5][0], "boot5_hi": boot[5][1],
                "boot10_lo": boot[10][0], "boot10_hi": boot[10][1],
                "boot20_lo": boot[20][0], "boot20_hi": boot[20][1],
                "boot10_excludes_zero": bool(boot[10][0] > 0 or boot[10][1] < 0),
            })
            print(f"  [{fam}] {task} {year}: mean {daily.mean():+.5f}  GW-uncond p={pu:.4f}  "
                  f"GW-cond p={pc:.4f}  boot10 [{boot[10][0]:+.5f},{boot[10][1]:+.5f}]",
                  flush=True)

    res = pd.DataFrame(rows)

    def bh(p: np.ndarray, q: float = 0.05) -> np.ndarray:
        order = np.argsort(p)
        n = len(p)
        crit = np.zeros(n, dtype=bool)
        passing = p[order] <= (np.arange(1, n + 1) / n) * q
        if passing.any():
            crit[order[:int(np.max(np.flatnonzero(passing))) + 1]] = True
        return crit

    # Benjamini-Hochberg WITHIN each family, for each test separately.
    for col in ("gw_uncond_p", "gw_cond_p"):
        flag = col.replace("_p", "_bh05")
        res[flag] = False
        for fam in res["family"].unique():
            m = (res["family"] == fam).to_numpy()
            res.loc[m, flag] = bh(res.loc[m, col].to_numpy())

    out = PROJECT_ROOT / "outputs" / "tables"
    res.to_csv(out / f"gw_inference{suffix}.csv", index=False)

    lines = ["# Forecast-comparison inference: Giacomini-White, block bootstrap, FDR\n",
             "Daily Brier loss differential, B minus A, so a POSITIVE mean favours B.\n",
             "  infoset : A = cal+AR(D-2), B = cal+AR(D-1)      -- the information-set question",
             "  forecast: A = cal+AR(D-1), B = cal+AR(D-1)+FC   -- the forecast-covariate question\n",
             res.round(5).to_markdown(index=False), ""]
    for fam in FAMILIES:
        r = res[res["family"] == fam]
        n = len(r)
        if not n:
            continue
        lines += [
            f"\n## Family `{fam}`: {n} target-year comparisons\n",
            f"- mean differential favours the richer specification in "
            f"**{int((r.mean_diff > 0).sum())} of {n}**",
            f"- Giacomini-White unconditional, individually significant at 5%: "
            f"**{int((r.gw_uncond_p < 0.05).sum())} of {n}**",
            f"- surviving Benjamini-Hochberg at FDR 5% within this family: "
            f"**{int(r.gw_uncond_bh05.sum())} of {n}**",
            f"- Giacomini-White conditional, individually significant at 5%: "
            f"**{int((r.gw_cond_p < 0.05).sum())} of {n}**; surviving BH: "
            f"**{int(r.gw_cond_bh05.sum())} of {n}**",
            f"- stationary bootstrap 95% interval excludes zero (mean block 10 days): "
            f"**{int(r.boot10_excludes_zero.sum())} of {n}**"]
    lines += ["\nBootstrap intervals at mean block lengths of 5, 10 and 20 days are reported in",
              "the table so that sensitivity to the block length can be judged directly."]
    text = "\n".join(lines)
    (PROJECT_ROOT / "experiments" / "16_inference" / f"FINDINGS{suffix}.md").write_text(
        text, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("\n" + "\n".join(lines[5:]))


if __name__ == "__main__":
    sys.exit(main())
