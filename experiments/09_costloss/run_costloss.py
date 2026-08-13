"""Cost-loss formalisation, risk-adjusted returns, and significance testing on the money.

Three gaps this closes, all identified by the key-paper structural analysis
(`literature/KEYPAPER_STRUCTURE.md`):

(1) THE 83% BREAK-EVEN PRECISION IS A COST-LOSS RATIO. Derive it, do not assert it.

    Write the decision problem as a regret matrix (regret against the best action in
    each state), for one MWh:

                          event occurs        event absent
        take action            0                   C
        no action              L                   0

    For the premium-eligible generator the event is "this hour is inside a >= 4 h
    negative block", the action is "curtail", C is the forfeited reference remuneration
    and L is the negative price actually paid when running through a suspended hour.
    For a flexible load the event is "this hour is negative", the action is "consume
    now", C is the shift cost and L is the forgone gain.

    With hit rate H, false-alarm rate F and base rate s, expected regret is
        E(H, F) = (1 - H) s L + F (1 - s) C
    so acting beats never acting iff  H s L > F (1 - s) C.  Since
        precision  pi = H s / (H s + F (1 - s)),
    that inequality is exactly
        pi / (1 - pi) > C / L = alpha        i.e.       pi > alpha / (1 + alpha).

    BREAK-EVEN PRECISION  pi* = alpha / (1 + alpha),  alpha = C / L.

    This is the cost-loss ratio of Murphy (1977) and Richardson (2000). It turns our
    headline into one line: the same classifier is worthless above pi* and valuable
    below it, and alpha is a property of the actor, not of the market.

(2) RELATIVE ECONOMIC VALUE AS A FUNCTION OF alpha (the Zhu et al. 2002 curve).
        V = 1 - E(H, F) / E_ref,     E_ref = min(s L, (1 - s) C),
    since perfect foresight has zero regret. V = 1 is perfect, V <= 0 is worthless.
    Sweeping alpha traces the value curve and shows where each actor sits on it.

(3) KATH & ZIEL'S REPORTING STANDARD. Their Table 4 carries a risk-adjusted column
    (Sharpe ratio) and Table 5 a significance test on the money, not only on the
    forecast error. We add both: daily P&L series -> mean/sd ratio, and a
    Diebold-Mariano test with the Harvey-Leybourne-Newbold correction on daily P&L
    differentials against the do-nothing benchmark.

Outputs
  outputs/tables/costloss_breakeven.csv   alpha, pi*, achieved precision, verdict
  outputs/tables/costloss_value_curve.csv V(alpha) for the figure
  outputs/tables/economic_significance.csv daily P&L stats + DM tests
  experiments/09_costloss/FINDINGS.md
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.features import (FEATURES_AR_D1, FEATURES_CAL, FEATURES_FC,  # noqa: E402
                          build, load_hourly)
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "08_infoset_dm"))
from run_infoset_dm import dm_hln  # noqa: E402

TEST_YEARS = [2021, 2022, 2023, 2024, 2025]
SEED = 42
# the corrected information set from experiment 08
FEATS = FEATURES_CAL + FEATURES_AR_D1 + FEATURES_FC
PREMIUM_LEVELS = (40.0, 60.0, 90.0)
SHIFT_COSTS = (0.0, 5.0, 15.0)
THR_GRID = np.append(np.linspace(0.05, 0.95, 19), 1.01)   # 1.01 == "never act"


def fit_predict(d, target, train_mask, pred_mask):
    X = d[FEATS]
    ok = X.notna().all(axis=1).to_numpy()
    tr, pr = train_mask & ok, pred_mask & ok
    m = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                       subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                       verbose=-1, n_jobs=4)
    m.fit(X[tr].to_numpy(), d[target].to_numpy()[tr])
    out = np.full(len(d), np.nan)
    out[pr] = m.predict_proba(X[pr].to_numpy())[:, 1]
    return out


def fit_predict_honest(d, target, tr, va, te):
    """Validation predictions must be OUT OF SAMPLE, or threshold tuning is meaningless.

    The first version of this script fitted a single model on `tr | va` and predicted on
    `va | te`. The validation-year predictions were therefore in-sample, and with 400 boosted
    trees on a rare event they were near-perfect: on 2023 the model flagged 259 hours with 259
    true positives and NO false positives at every threshold in the grid. Every threshold then
    looked equally good, the tuner returned the first grid point (0.05), and that far-too-
    aggressive rule was applied to genuinely out-of-sample test data. The realised losses that
    followed were an artefact of the procedure, not of the economics.

    The corrected design:
      - tune the acting threshold on validation predictions from a model fitted on `tr` only;
      - refit on `tr | va` and apply that threshold to the test year.
    This is the standard train / validate / test split and it is what Section 5.4 describes.
    """
    p_va = fit_predict(d, target, tr, va)
    p_te = fit_predict(d, target, tr | va, te)
    out = np.full(len(d), np.nan)
    out[va] = p_va[va]
    out[te] = p_te[te]
    return out


# ------------------------------------------------------------------ cost-loss algebra
def breakeven_precision(alpha: float) -> float:
    return alpha / (1.0 + alpha)


def relative_value(H: float, F: float, s: float, alpha: float) -> float:
    """V = 1 - E(H,F)/E_ref, in units of L (so only alpha = C/L matters)."""
    e = (1 - H) * s + F * (1 - s) * alpha
    e_ref = min(s, (1 - s) * alpha)
    return float(1 - e / e_ref) if e_ref > 0 else np.nan


def rates(y: np.ndarray, act: np.ndarray) -> dict:
    y = y.astype(bool)
    act = act.astype(bool)
    tp = int((y & act).sum()); fp = int((~y & act).sum())
    fn = int((y & ~act).sum()); tn = int((~y & ~act).sum())
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "H": tp / (tp + fn) if tp + fn else np.nan,
            "F": fp / (fp + tn) if fp + tn else np.nan,
            "precision": tp / (tp + fp) if tp + fp else np.nan,
            "base_rate": float(y.mean())}


# ------------------------------------------------------------------ P&L (per MW-hour)
def gen_pnl(act, price, suspended, weight, premium):
    """Hourly revenue for a premium generator; `act` = curtail."""
    per_hour = np.where(suspended, price, premium)
    return (~act) * weight * per_hour


def flex_pnl(act, price, shift_cost):
    """Hourly value for a flexible load; `act` = move the block into this hour."""
    return act * (-price - shift_cost)


def tune(pnl_fn, p, grid=THR_GRID):
    best_t, best_v = 1.01, -np.inf
    for t in grid:
        v = float(np.sum(pnl_fn(p >= t)))
        if v > best_v:
            best_t, best_v = float(t), v
    return best_t


def incremental(da: np.ndarray, dn: np.ndarray) -> dict:
    """Statistics of the INCREMENTAL daily P&L (acting minus benchmark).

    Kath & Ziel (2018) report a Sharpe ratio on the portfolio price level. For a
    curtailment decision the level is dominated by the premium and is nearly identical
    across rules, so the informative quantity is the increment: mean/sd of (act - nil).
    Both are reported so the two are not confused.
    """
    inc = da - dn
    sd_i = float(inc.std(ddof=1))
    return {"annual_gain": float(inc.sum()),
            "mean_daily_inc": float(inc.mean()), "sd_daily_inc": sd_i,
            "sharpe_incremental": float(inc.mean() / sd_i) if sd_i > 0 else np.nan,
            "mean_daily_level": float(da.mean()),
            "sharpe_level": (float(da.mean() / da.std(ddof=1))
                             if da.std(ddof=1) > 0 else np.nan)}


def daily(series: np.ndarray, index: pd.DatetimeIndex) -> np.ndarray:
    s = pd.Series(series, index=index)
    return s.groupby(s.index.normalize()).sum().to_numpy()


def main() -> None:
    d, _ = build(load_hourly())
    d["avail"] = (d["res_fc"] / d["res_fc"].max()).clip(0, 1)
    price = d["price"].to_numpy()
    suspended = d["y_run4"].to_numpy().astype(bool)
    neg = d["y_neg"].to_numpy().astype(bool)
    w = d["avail"].to_numpy()
    years = d.index.year

    be_rows, sig_rows = [], []

    for ty in TEST_YEARS:
        tr, va, te = years < ty - 1, years == ty - 1, years == ty
        if tr.sum() < 2000 or va.sum() < 500 or te.sum() < 500:
            continue
        p_neg = fit_predict_honest(d, "y_neg", tr, va, te)
        p_run = fit_predict_honest(d, "y_run4", tr, va, te)
        for nm, p in (("y_neg", p_neg), ("y_run4", p_run)):
            a = p[va] >= 0.5
            yv = d[nm].to_numpy()[va].astype(bool)
            print(f"    {ty} val {nm}: acted {a.sum()}, TP {(a & yv).sum()}, "
                  f"FP {(a & ~yv).sum()}  (FP must be > 0, or tuning is in-sample)",
                  flush=True)
        idx_te = d.index[te]

        # ---------------- generator ------------------------------------------
        for premium in PREMIUM_LEVELS:
            t = tune(lambda a: gen_pnl(a, price[va], suspended[va], w[va], premium), p_run[va])
            act_te = p_run[te] >= t
            r = rates(suspended[te], act_te)
            # C = forfeited remuneration, L = mean magnitude of the negative price paid
            # L must be knowable at the time the decision rule is set, so it is estimated on
            # everything BEFORE the test year. The first version used the test year's own
            # realised negative prices, which put evaluation-period data inside the criterion
            # and made L swing from 2.37 to 18.58 EUR/MWh across years for a fixed contract.
            pre = tr | va
            sus_pre = price[pre][suspended[pre] & (price[pre] < 0)]
            if sus_pre.size < 50:
                print(f"    {ty} premium {premium:g}: too few prior suspended hours, skipped")
                continue
            L = float(np.abs(sus_pre).mean())
            alpha = premium / L
            pnl_act = gen_pnl(act_te, price[te], suspended[te], w[te], premium)
            pnl_nil = gen_pnl(np.zeros(te.sum(), bool), price[te], suspended[te], w[te], premium)
            da, dn = daily(pnl_act, idx_te), daily(pnl_nil, idx_te)
            dm = dm_hln(-da, -dn)      # loss = -P&L; negative DM => acting is better
            be_rows.append({"actor": "premium_generator", "test_year": ty, "param": premium,
                            "C": premium, "L": round(L, 3), "alpha": round(alpha, 3),
                            "breakeven_precision": round(breakeven_precision(alpha), 4),
                            "achieved_precision": r["precision"], "threshold": t,
                            "H": r["H"], "F": r["F"], "base_rate": r["base_rate"],
                            "V": relative_value(r["H"], r["F"], r["base_rate"], alpha),
                            "pays": bool(pd.notna(r["precision"])
                                         and r["precision"] > breakeven_precision(alpha))})
            sig_rows.append({"actor": "premium_generator", "test_year": ty, "param": premium,
                             **incremental(da, dn), "DM": dm["DM"],
                             "p_value": dm["p_value"], "T": dm["T"],
                             "TP": r["TP"], "FP": r["FP"], "FN": r["FN"], "TN": r["TN"],
                             "L_year": round(L, 3)})

        # ---------------- flexible load --------------------------------------
        for sc in SHIFT_COSTS:
            t = tune(lambda a: flex_pnl(a, price[va], sc), p_neg[va])
            act_te = p_neg[te] >= t
            r = rates(neg[te], act_te)
            L = float(np.abs(price[te][neg[te]]).mean())
            alpha = sc / L
            pnl_act = flex_pnl(act_te, price[te], sc)
            pnl_nil = np.zeros(te.sum())
            da, dn = daily(pnl_act, idx_te), daily(pnl_nil, idx_te)
            dm = dm_hln(-da, -dn)
            be_rows.append({"actor": "flexible_load", "test_year": ty, "param": sc,
                            "C": sc, "L": round(L, 3), "alpha": round(alpha, 3),
                            "breakeven_precision": round(breakeven_precision(alpha), 4),
                            "achieved_precision": r["precision"], "threshold": t,
                            "H": r["H"], "F": r["F"], "base_rate": r["base_rate"],
                            "V": relative_value(r["H"], r["F"], r["base_rate"], alpha)
                                 if alpha > 0 else np.nan,
                            "pays": bool(pd.notna(r["precision"])
                                         and r["precision"] > breakeven_precision(alpha))})
            sig_rows.append({"actor": "flexible_load", "test_year": ty, "param": sc,
                             **incremental(da, dn), "DM": dm["DM"],
                             "p_value": dm["p_value"], "T": dm["T"],
                             "TP": r["TP"], "FP": r["FP"], "FN": r["FN"], "TN": r["TN"],
                             "L_year": round(L, 3)})
        print(f"  {ty} done", flush=True)

    be = pd.DataFrame(be_rows)
    sig = pd.DataFrame(sig_rows)
    out = PROJECT_ROOT / "outputs" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    be.to_csv(out / "costloss_breakeven.csv", index=False)
    sig.to_csv(out / "economic_significance.csv", index=False)

    # ---------------- the value curve (one row per alpha per actor/year) -------
    alphas = np.round(np.concatenate([np.linspace(0.02, 1.0, 50),
                                      np.linspace(1.1, 8.0, 70)]), 4)
    curve = []
    for _, r in be.iterrows():
        if not np.isfinite(r["H"]) or not np.isfinite(r["F"]):
            continue
        for a in alphas:
            curve.append({"actor": r["actor"], "test_year": r["test_year"],
                          "param": r["param"], "alpha": a,
                          "V": relative_value(r["H"], r["F"], r["base_rate"], a)})
    pd.DataFrame(curve).to_csv(out / "costloss_value_curve.csv", index=False)

    # ---------------- report ---------------------------------------------------
    # POOLED across test years. Averaging per-year alpha would be a Jensen error: alpha is a
    # ratio, so it must be formed from pooled C and pooled L, and precision from the pooled
    # contingency table, not from a mean of yearly precisions.
    pooled = (sig.groupby(["actor", "param"])[["TP", "FP", "FN", "TN"]].sum()
              .join(be.groupby(["actor", "param"])["L"].mean().rename("L_mean")))
    Lp = {}
    for (actor, param), _ in pooled.iterrows():
        sub = sig[(sig.actor == actor) & (sig.param == param)]
        Lp[(actor, param)] = float(sub["L_year"].mean())
    pooled["L"] = [Lp[i] for i in pooled.index]
    pooled["C"] = [i[1] for i in pooled.index]
    pooled["alpha"] = pooled["C"] / pooled["L"]
    pooled["pistar"] = pooled["alpha"] / (1 + pooled["alpha"])
    pooled["n_acted"] = pooled["TP"] + pooled["FP"]
    pooled["pi"] = pooled["TP"] / pooled["n_acted"].replace(0, np.nan)
    # 95% Jeffreys interval for the binomial proportion. Precision is estimated from a few
    # hundred acted-upon hours, so the comparison with pi* is not resolved by the point
    # estimate alone; Section 7.4 reports which rows the interval actually separates.
    from scipy import stats as _st
    lo, hi = _st.beta.ppf([[0.025], [0.975]],
                          pooled["TP"] + 0.5, pooled["FP"] + 0.5)
    pooled["pi_lo"], pooled["pi_hi"] = lo, hi
    pooled["resolves"] = np.where(pooled["pi_lo"] > pooled["pistar"], "above",
                          np.where(pooled["pi_hi"] < pooled["pistar"], "below", "unresolved"))
    # the crossing point: the largest cost-loss ratio at which this classifier still pays
    pooled["alpha_star"] = pooled["pi"] / (1 - pooled["pi"])
    pooled = pooled.reset_index()
    pooled.to_csv(out / "costloss_pooled.csv", index=False)

    lines = ["# Cost-Loss Formalisation, Risk Adjustment and Economic Significance\n",
             "Break-even precision is **derived**, not asserted: `pi* = alpha/(1+alpha)` with",
             "`alpha = C/L`. See the module docstring for the two-line proof.\n",
             "\n## Break-even vs achieved precision (pooled over the five test years)\n",
             "`alpha` and `pi` are formed from pooled quantities, never averaged across years",
             "(a ratio of means, not a mean of ratios).\n",
             "| actor | C | L | alpha = C/L | break-even pi* | achieved pi | pays? |",
             "|---|---:|---:|---:|---:|---:|---|"]
    yearly_pays = be.groupby(["actor", "param"])["pays"].agg(["sum", "size"])
    for _, r in pooled.iterrows():
        pi = f"{r.pi:.3f}" if pd.notna(r.pi) else "n/a (never acts)"
        s, n = yearly_pays.loc[(r.actor, r.param)]
        lines.append(f"| {r.actor} (param {r['param']:g}) | {r.C:.2f} | {r.L:.2f} | "
                     f"{r.alpha:.2f} | {r.pistar:.3f} | {pi} | {int(s)}/{int(n)} years |")

    astar = pooled["alpha_star"].dropna()
    if len(astar):
        lines += ["", f"\n**The single number that decides everything.** The classifier's pooled "
                      f"precision implies a crossing point `alpha* = pi/(1-pi)` of "
                      f"**{astar.min():.2f} to {astar.max():.2f}**. A decision-maker whose "
                      "cost-loss ratio is below that band profits from exactly the same "
                      "forecast that is worthless to one above it. The sign of the asymmetry "
                      "is a property of the actor, not of the market."]

    lines += ["\n\n## Economic significance (daily P&L, DM with HLN correction)\n",
              "Negative DM => acting beats the benchmark. `sharpe_inc` is the mean/sd of the",
              "INCREMENTAL daily P&L (act minus benchmark); `sharpe_lvl` is the level Sharpe",
              "reported by Kath & Ziel, which for curtailment is dominated by the premium and",
              "is therefore nearly constant across rules.\n",
              "| actor | param | year | annual gain | sharpe_inc | sharpe_lvl | DM | p |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in sig.iterrows():
        f2 = lambda v: f"{v:.3f}" if pd.notna(v) else "n/a"  # noqa: E731
        lines.append(f"| {r.actor} | {r['param']:g} | {int(r.test_year)} | "
                     f"{r.annual_gain:+,.0f} | {f2(r.sharpe_incremental)} | "
                     f"{f2(r.sharpe_level)} | "
                     f"{f'{r.DM:.2f}' if pd.notna(r.DM) else 'n/a'} | "
                     f"{f'{r.p_value:.4f}' if pd.notna(r.p_value) else 'n/a'} |")

    text = "\n".join(lines)
    (PROJECT_ROOT / "experiments" / "09_costloss").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "experiments" / "09_costloss" / "FINDINGS.md").write_text(
        text, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(text)


if __name__ == "__main__":
    sys.exit(main())
