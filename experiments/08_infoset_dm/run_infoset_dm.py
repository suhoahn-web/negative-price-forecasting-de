"""Information-set correction + Diebold-Mariano testing for the baseline suite.

Two things this fixes, both raised by the Lago et al. (2021, Applied Energy 293:116983)
checklist for adequate EPF research:

(1) THE AUTOREGRESSIVE BENCHMARK WAS HANDICAPPED.
    `src/features.py` originally used only D-2 price lags, described as "conservative".
    It is conservative for OUR model but it silently weakens the benchmark we compare
    against, which inflates the reported gain from day-ahead forecast features.
    The day-ahead auction for delivery day D-1 clears at 12:00 CET on D-2, so at gate
    closure for day D (12:00 CET on D-1) the entire D-1 price vector is public.
    A fair autoregressive benchmark is entitled to it. Here we measure the difference
    instead of assuming it away.

    PRE-REGISTERED READING OF THE OUTCOME, fixed before the run:
      - If the forecast-feature gain over the CORRECTED (D-1) autoregressive benchmark
        stays positive in >= 4 of 5 test years on both hourly tasks, the headline claim
        survives and we report the corrected number as the headline.
      - If it does not, the headline claim was an artefact of the handicap and the paper
        must be rewritten around it. Either way the D-2 vs D-1 comparison is reported.

(2) NO SIGNIFICANCE TESTING.
    Lago et al.: significance testing "is seldom conducted". We add Diebold-Mariano
    (1995) with the Harvey-Leybourne-Newbold (1997) small-sample correction.

    Applying DM to a CLASSIFICATION problem needs care, so:
      - PR-AUC is a ranking statistic over the whole sample and has no per-observation
        decomposition, so it cannot be DM-tested. It stays the headline metric and is
        reported without a test.
      - The test is run on the BRIER SCORE, which is a strictly proper scoring rule
        (Gneiting & Raftery 2007) and decomposes per observation. Log-loss is reported
        alongside as a robustness check.
      - Day-ahead forecasts for the 24 hours of a day are issued jointly, so hourly loss
        differentials are strongly correlated within a day. Following the multivariate DM
        practice of Lago et al., the loss is aggregated to ONE VALUE PER DAY before
        testing. A Newey-West HAC variance is used on top, in case daily differentials
        remain autocorrelated.

Outputs
  outputs/tables/infoset_comparison.csv   per task/year/model/featureset scores + runtime
  outputs/preds/<task>_<year>.parquet     per-observation probabilities (for reuse)
  outputs/tables/dm_tests.csv             DM/HLN statistics and p-values
  experiments/08_infoset_dm/FINDINGS.md   written by write_findings.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.features import (FEATURES_AR, FEATURES_AR_D1, FEATURES_CAL,  # noqa: E402
                          FEATURES_FC, build, load_hourly)

TEST_YEARS = [2021, 2022, 2023, 2024, 2025]
SEED = 42
EPS = 1e-6

FEATURE_SETS = {
    "cal":            FEATURES_CAL,
    "cal+AR(D-2)":    FEATURES_CAL + FEATURES_AR,
    "cal+AR(D-1)":    FEATURES_CAL + FEATURES_AR_D1,
    "cal+FC":         FEATURES_CAL + FEATURES_FC,
    "cal+AR(D-1)+FC": FEATURES_CAL + FEATURES_AR_D1 + FEATURES_FC,
}
# the two families the headline claim compares
AR_SETS = ["cal+AR(D-2)", "cal+AR(D-1)"]
FC_SETS = ["cal+FC", "cal+AR(D-1)+FC"]


# --------------------------------------------------------------------------- metrics
def score(y, p) -> dict:
    return {"PR_AUC": float(average_precision_score(y, p)),
            "ROC_AUC": float(roc_auc_score(y, p)),
            "Brier": float(brier_score_loss(y, p)),
            "LogLoss": float(-np.mean(y * np.log(np.clip(p, EPS, 1 - EPS))
                                      + (1 - y) * np.log(np.clip(1 - p, EPS, 1 - EPS)))),
            "base_rate": float(np.mean(y))}


# --------------------------------------------------------------------------- models
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


MODELS = {"logit": fit_logit, "lgbm": fit_lgbm}


# --------------------------------------------------------------------------- DM test
def dm_hln(loss_a: np.ndarray, loss_b: np.ndarray, h: int = 1) -> dict:
    """Diebold-Mariano on the differential d = loss_a - loss_b, HLN-corrected.

    Negative statistic => model A has the LOWER loss => A is better.
    Variance uses a Newey-West HAC estimator with the Newey-West bandwidth rule; for h=1
    and uncorrelated differentials this reduces to the plain sample variance.
    Two-sided p-value from t(T-1), per Harvey, Leybourne & Newbold (1997).
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = d[np.isfinite(d)]
    T = d.size
    if T < 30:
        return {"T": T, "mean_diff": float(np.mean(d)) if T else np.nan,
                "DM": np.nan, "p_value": np.nan, "hac_lags": np.nan}
    dbar = float(np.mean(d))
    dc = d - dbar
    L = max(h - 1, int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0))))
    gamma0 = float(dc @ dc) / T
    var = gamma0
    for k in range(1, L + 1):
        gk = float(dc[k:] @ dc[:-k]) / T
        var += 2.0 * (1.0 - k / (L + 1.0)) * gk      # Bartlett kernel
    var = max(var, 1e-18)
    dm = dbar / np.sqrt(var / T)
    corr = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)   # HLN factor
    dm_hln_stat = dm * corr
    p = float(2 * (1 - stats.t.cdf(abs(dm_hln_stat), df=T - 1)))
    return {"T": T, "mean_diff": dbar, "DM": float(dm_hln_stat),
            "p_value": p, "hac_lags": L}


# --------------------------------------------------------------------------- runner
def prepare(d: pd.DataFrame, target: str, level: str) -> pd.DataFrame:
    if level == "hour":
        return d
    cols = FEATURES_CAL + FEATURES_AR_D1 + FEATURES_FC
    agg = {c: "mean" for c in cols if c in d.columns}
    extra = d.groupby("day").agg(
        resload_min=("residual_load", "min"),
        res_share_max=("res_share", "max"),
        n_low_hours=("residual_load", lambda s: int((s < s.quantile(0.25)).sum())))
    frame = d.groupby("day").agg(agg).join(extra)
    frame[target] = d.groupby("day")[target].max()
    frame.index = pd.to_datetime(frame.index)
    return frame


def fit_block(fn, X, y_all, tr_mask, te_mask, recal: str, frame, test_year, min_train,
              window_days: int | None = None):
    """Fit once per year, or refit each month.

    `window_days` controls the MEMORY of the estimator, which is a separate choice from the
    refit schedule and matters for which inference is licensed. With None the training set is
    everything preceding the cutoff, an expanding window. With an integer the training set is
    only the last `window_days` days before the cutoff, a rolling window of fixed length.

    Giacomini & White (2006) require the maximum estimation sample size to stay finite as the
    out-of-sample count grows -- their Comment 2 is explicit that "the asymptotic distribution
    is obtained for the number of out-of-sample observations n going to infinity, whereas the
    maximum estimation sample size m is finite". An expanding window does not satisfy this, so
    a GW test run on expanding-window forecasts is outside the theory. The rolling variant
    exists so the test has forecasts it is actually valid for.
    """
    if recal == "year":
        return fn(X[tr_mask].to_numpy(), y_all[tr_mask].to_numpy(), X[te_mask].to_numpy())
    out = np.full(int(te_mask.sum()), np.nan)
    te_idx = frame.index[te_mask]
    for mo in range(1, 13):
        blk = te_idx.month == mo
        if not blk.any():
            continue
        cutoff = pd.Timestamp(year=test_year, month=mo, day=1, tz=frame.index.tz)
        tr = np.asarray(frame.index < cutoff) & (tr_mask | te_mask)
        if window_days is not None:
            start = cutoff - pd.Timedelta(days=window_days)
            tr &= np.asarray(frame.index >= start)
        if tr.sum() < min_train or y_all[tr].sum() < 20:
            continue
        sub = te_mask.copy()
        sub[te_mask] = blk
        out[blk] = fn(X[tr].to_numpy(), y_all[tr].to_numpy(), X[sub].to_numpy())
    return out


def run_task(d: pd.DataFrame, target: str, level: str, rows: list, preds_out: Path,
             recal: str = "year", window_days: int | None = None) -> None:
    frame = prepare(d, target, level)
    y_all = frame[target]
    min_train, min_test = (2000, 100) if level == "hour" else (500, 60)

    for test_year in TEST_YEARS:
        tr = frame.index.year < test_year
        te = frame.index.year == test_year
        if tr.sum() < min_train or te.sum() < min_test:
            continue
        base = float(y_all[te].mean())
        if base <= 0:
            continue

        store = {"y": y_all[te].to_numpy()}
        store_index = frame.index[te]

        for fname, cols in FEATURE_SETS.items():
            cols = [c for c in cols if c in frame.columns]
            X = frame[cols]
            ok_tr = tr & X.notna().all(axis=1).to_numpy()
            ok_te = te & X.notna().all(axis=1).to_numpy()
            if ok_tr.sum() < min_train or ok_te.sum() < min_test or y_all[ok_tr].sum() < 20:
                print(f"    skip {fname} {test_year} (train {ok_tr.sum()}, test {ok_te.sum()})")
                continue
            for mname, fn in MODELS.items():
                t0 = time.perf_counter()
                p = fit_block(fn, X, y_all, ok_tr, ok_te, recal, frame, test_year, min_train,
                              window_days)
                fit_s = time.perf_counter() - t0
                keep = np.isfinite(p)
                if keep.sum() < min_test:
                    print(f"    skip {fname}/{mname} {test_year}: only {keep.sum()} scored")
                    continue
                rows.append({"task": target, "level": level, "test_year": test_year,
                             "features": fname, "model": mname, "recal": recal,
                             "n_train": int(ok_tr.sum()), "n_test": int(ok_te.sum()),
                             "n_features": len(cols), "fit_seconds": round(fit_s, 3),
                             **score(y_all[ok_te].to_numpy()[keep], p[keep])})
                # align predictions onto the full test index (NaN where features were missing)
                col = np.full(te.sum(), np.nan)
                pos = np.flatnonzero(ok_te[te])
                col[pos] = p
                store[f"{mname}|{fname}"] = col
            print(f"  {target} {test_year} {fname:16s} done", flush=True)

        pd.DataFrame(store, index=store_index).to_parquet(
            preds_out / f"{target}_{test_year}.parquet")


# --------------------------------------------------------------------------- main
def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--recal", choices=["year", "month"], default="year",
                    help="refit once per test year, or monthly on all preceding data "
                         "(Section 5.4 adopts monthly; see experiment 11)")
    ap.add_argument("--window-days", type=int, default=None,
                    help="cap the training window at this many days before each refit "
                         "cutoff. Required for the Giacomini-White test to apply; see "
                         "fit_block.")
    args = ap.parse_args()
    sfx = "" if args.recal == "year" else "_monthly"
    if args.window_days:
        sfx += f"_w{args.window_days}"

    out_t = PROJECT_ROOT / "outputs" / "tables"
    sub = "" if args.recal == "year" else "monthly"
    if args.window_days:
        sub = f"rolling{args.window_days}"
    out_p = PROJECT_ROOT / "outputs" / "preds" / sub
    out_t.mkdir(parents=True, exist_ok=True)
    out_p.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    df = load_hourly()
    d, _ = build(df)
    print("panel:", d.shape, d.index.min(), "->", d.index.max(), flush=True)
    for c in ("price_h_d1", "d1_max_run", "d1_n_neg"):
        print(f"  {c}: {d[c].notna().mean():.3f} non-null, "
              f"mean {d[c].mean():.3f}", flush=True)

    rows: list = []
    # every run length § 51 EEG has used: six for the 2016-20 cohort, four for 2021-22,
    # three for the 2023 cohort from 2024, and none at all from 25 Feb 2025 (= y_neg)
    for tgt in ("y_neg", "y_run3", "y_run4", "y_run6"):
        run_task(d, tgt, "hour", rows, out_p, args.recal, args.window_days)
    run_task(d, "y_day4", "day", rows, out_p, args.recal, args.window_days)

    res = pd.DataFrame(rows)
    res.to_csv(out_t / f"infoset_comparison{sfx}.csv", index=False)
    print(f"\nwrote {len(res)} rows; total {time.perf_counter() - t_start:.1f}s", flush=True)

    # ---------------- DM tests -------------------------------------------------
    dm_rows: list = []
    for task, level in (("y_neg", "hour"), ("y_run3", "hour"), ("y_run4", "hour"),
                        ("y_run6", "hour"), ("y_day4", "day")):
        for year in TEST_YEARS:
            f = out_p / f"{task}_{year}.parquet"
            if not f.exists():
                continue
            P = pd.read_parquet(f)
            y = P["y"].to_numpy(float)
            cols = [c for c in P.columns if c != "y"]

            def daily_loss(p, kind="brier"):
                p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
                if kind == "brier":
                    ell = (p - y) ** 2
                else:
                    ell = -(y * np.log(p) + (1 - y) * np.log(1 - p))
                s = pd.Series(ell, index=P.index)
                return s.groupby(s.index.normalize()).mean() if level == "hour" else s

            # every pairwise comparison inside the same model family
            for kind in ("brier", "logloss"):
                for i, ca in enumerate(cols):
                    for cb in cols[i + 1:]:
                        ma, fa = ca.split("|")
                        mb, fb = cb.split("|")
                        if ma != mb:          # compare feature sets, holding the model fixed
                            continue
                        la, lb = daily_loss(P[ca], kind), daily_loss(P[cb], kind)
                        ok = la.notna() & lb.notna()
                        r = dm_hln(la[ok].to_numpy(), lb[ok].to_numpy())
                        dm_rows.append({"task": task, "test_year": year, "loss": kind,
                                        "model": ma, "A": fa, "B": fb, **r})

    dm = pd.DataFrame(dm_rows)
    dm.to_csv(out_t / f"dm_tests{sfx}.csv", index=False)
    print(f"wrote {len(dm)} DM tests", flush=True)

    # ---------------- the pre-registered headline check ------------------------
    print("\n=== PRE-REGISTERED CHECK: does the forecast gain survive a fair AR benchmark? ===")
    for task in ("y_neg", "y_run4", "y_day4"):
        sub = res[res.task == task]
        if sub.empty:
            continue
        print(f"\n{task}")
        for label, sets in (("AR(D-2) [old]", ["cal+AR(D-2)"]),
                            ("AR(D-1) [fair]", ["cal+AR(D-1)"]),
                            ("with forecasts", FC_SETS)):
            best = sub[sub.features.isin(sets)].groupby("test_year")["PR_AUC"].max()
            print(f"  {label:16s} " + " ".join(f"{y}:{v:.3f}" for y, v in best.items()))
        old = sub[sub.features.isin(["cal+AR(D-2)"])].groupby("test_year")["PR_AUC"].max()
        fair = sub[sub.features.isin(["cal+AR(D-1)"])].groupby("test_year")["PR_AUC"].max()
        fc = sub[sub.features.isin(FC_SETS)].groupby("test_year")["PR_AUC"].max()
        g_old = (100 * (fc - old) / old).dropna()
        g_fair = (100 * (fc - fair) / fair).dropna()
        print(f"  gain vs OLD  benchmark: " + " ".join(f"{y}:{v:+.1f}%" for y, v in g_old.items()))
        print(f"  gain vs FAIR benchmark: " + " ".join(f"{y}:{v:+.1f}%" for y, v in g_fair.items()))
        print(f"  years positive vs fair benchmark: {(g_fair > 0).sum()} / {len(g_fair)}")


if __name__ == "__main__":
    sys.exit(main())
