"""A load-shifting decision with an energy balance, a displaced counterfactual and an
activation limit.

WHY THIS REPLACES THE EARLIER FLEXIBLE-LOAD MODEL
The first version valued the load's decision as `act * (-price - shift_cost)`: it counted the
gain in every hour the rule fired and never charged for the energy that had to be given up
somewhere else. Three things were wrong with it. There was no energy balance, so consumption
was created rather than moved. There was no displaced counterfactual, so the benefit was
measured against zero instead of against the hour the load would otherwise have used. And there
was no activation limit, so the "flexible block" fired in 422 separate hours of 2025, which is
not a block. With the generator demoted to a counterexample (Section 7.2), this is now the only
evaluated decision in the paper and it has to be right.

THE DECISION MODELLED
A load must run a process for N hours of every day and must fix the schedule in the morning of
D-1, before gate closure. Energy is conserved by construction: exactly N hours are chosen every
day, so any hour brought forward displaces one that is given up.

  baseline    the N hours with the lowest climatological price for that month and hour,
              estimated on training data only. This is what an operator does with no forecast:
              run at the historically cheapest times.
  forecast    rank the 24 hours by the classifier's predicted probability that the price is
              negative, and take the N highest. Deviate from the baseline only if the Nth
              ranked score exceeds a threshold tuned on the PRECEDING year, and the tuning grid
              contains a never-act option.

  value(day) = sum(price over baseline hours) - sum(price over chosen hours)
               - shift_cost * (number of hours changed)

Every term is a real transfer. The first two differ only in which hours are used, so the energy
balance holds exactly. The third charges for rescheduling.

COST AND LOSS, MEASURED RATHER THAN ASSUMED
The regret of a marginal substitution is heterogeneous, so C and L are reported as sample means
over the substitutions actually made:
  L = E[ p(displaced) - p(chosen) | chosen hour negative ]      gain forgone by not moving
  C = E[ p(chosen) - p(displaced) | chosen hour not negative ] + shift cost
This makes alpha = C/L partly a market quantity and partly a contract quantity; Section 7.3
must say so. Holding the market fixed and varying the shift cost still isolates the contract.

Predictions come from `outputs/preds/monthly/`, which are the monthly-refit out-of-sample
probabilities of Section 5.4, so no model is fitted here and no validation year is reused.

Outputs
  outputs/tables/load_shifting.csv
  experiments/14_load_shifting/FINDINGS.md
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.features import build, load_hourly  # noqa: E402
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "08_infoset_dm"))
from run_infoset_dm import dm_hln  # noqa: E402

TEST_YEARS = [2021, 2022, 2023, 2024, 2025]
N_HOURS = 4                       # length of the daily production block
SHIFT_COSTS = (0.0, 5.0, 15.0)
GRID = np.append(np.linspace(0.05, 0.95, 19), 1.01)   # 1.01 = never deviate
PRED_COL = "lgbm|cal+AR(D-1)+FC"


def load_predictions() -> pd.Series:
    out = []
    for f in sorted((PROJECT_ROOT / "outputs" / "preds" / "monthly").glob("y_neg_*.parquet")):
        df = pd.read_parquet(f)
        if PRED_COL in df.columns:
            out.append(df[PRED_COL].dropna())
    return pd.concat(out).sort_index()


def climatological_baseline(price: pd.Series, train_mask: np.ndarray) -> pd.Series:
    """For each (month, hour) the mean training price; used to rank hours with no forecast."""
    tr = price[train_mask]
    key = pd.MultiIndex.from_arrays([tr.index.month, tr.index.hour])
    clim = tr.groupby(key).mean()
    full_key = pd.MultiIndex.from_arrays([price.index.month, price.index.hour])
    return pd.Series(clim.reindex(full_key).to_numpy(), index=price.index)


def day_value(prices, clim_rank, score, n, thr, shift_cost):
    """Value of one day's schedule, in EUR per MWh of block, against the climatological plan."""
    base = np.argsort(clim_rank)[:n]                 # cheapest n hours climatologically
    if thr > 1.0 or np.sort(score)[::-1][n - 1] < thr:
        return 0.0, 0                                 # do not deviate
    chosen = np.argsort(-score)[:n]
    changed = len(set(chosen.tolist()) - set(base.tolist()))
    val = prices[base].sum() - prices[chosen].sum() - shift_cost * changed
    return float(val), changed


def evaluate_year(day_frames, thr, shift_cost):
    vals, changes = [], 0
    for pr, cl, sc in day_frames:
        v, c = day_value(pr, cl, sc, N_HOURS, thr, shift_cost)
        vals.append(v)
        changes += c
    return np.array(vals), changes


def substitution_stats(day_frames, thr, shift_cost):
    """Sample C and L for the BINARY decision the load actually faces.

    The load's payoff is continuous in price, so "the price is negative" is not the event its
    payoff turns on; it is a signal the load uses. The binary decision is whether to deviate
    from the climatological baseline on a given day, and the event that makes deviating correct
    is that the deviation turns out to pay after the shift cost. Defining C and L against the
    sign of the price instead produced negative cost-loss ratios, because moving into an hour
    that is merely cheap rather than negative is not a loss. This is the decision-relevant
    event, and it keeps alpha positive by construction.

      L = mean realised gain on days when deviating paid
      C = mean realised loss on days when it did not
      precision = share of deviating days that paid
    """
    gains, losses = [], []
    for pr, cl, sc in day_frames:
        v, _ = day_value(pr, cl, sc, N_HOURS, thr, shift_cost)
        if thr > 1.0 or np.sort(sc)[::-1][N_HOURS - 1] < thr:
            continue
        (gains if v > 0 else losses).append(v)
    L = float(np.mean(gains)) if gains else np.nan
    C = -float(np.mean(losses)) if losses else np.nan
    return C, L, len(gains), len(losses)


def main() -> None:
    d, _ = build(load_hourly())
    price = d["price"]
    neg = (d["y_neg"] == 1)
    pred = load_predictions()
    idx = price.index.intersection(pred.index)
    price, neg, pred = price.loc[idx], neg.loc[idx], pred.loc[idx]
    print(f"aligned {len(idx):,} hours, {idx.min()} -> {idx.max()}", flush=True)

    rows = []
    for ty in TEST_YEARS:
        va_mask = np.asarray(price.index.year == ty - 1)
        te_mask = np.asarray(price.index.year == ty)
        if va_mask.sum() < 5000 or te_mask.sum() < 5000:
            print(f"  {ty}: insufficient validation or test coverage, skipped")
            continue
        clim = climatological_baseline(price, np.asarray(price.index.year < ty))

        def frames(mask):
            df = pd.DataFrame({"p": price[mask], "c": clim[mask], "s": pred[mask],
                               "n": neg[mask]})
            out, negs = [], []
            for _, g in df.groupby(df.index.normalize()):
                if len(g) != 24:
                    continue
                out.append((g["p"].to_numpy(), g["c"].to_numpy(), g["s"].to_numpy()))
                negs.append(g["n"].to_numpy())
            return out, negs

        va, _ = frames(va_mask)
        te, te_neg = frames(te_mask)

        for sc_cost in SHIFT_COSTS:
            best_t, best_v = 1.01, -np.inf
            for t in GRID:
                v, _ = evaluate_year(va, t, sc_cost)
                if v.sum() > best_v:
                    best_t, best_v = float(t), v.sum()
            vals, changed = evaluate_year(te, best_t, sc_cost)
            C, L, n_gain, n_cost = substitution_stats(te, best_t, sc_cost)
            # C, L and pi are all realised test-year quantities, so comparing pi with
            # pi* = alpha/(1+alpha) on the SAME year is an accounting identity, not a test:
            # value = n_g*L - n_l*C > 0 iff n_g/n_l > C/L iff pi > pi*. It shows why the rule
            # pays; it cannot show that the criterion has predictive content. The genuine test
            # is whether alpha estimated on the PRECEDING year forecasts the sign of the next
            # year's value, which is what alpha_exante below records.
            C_ex, L_ex, ng_ex, nl_ex = substitution_stats(va, best_t, sc_cost)
            alpha_ex = (C_ex / L_ex) if (L_ex and np.isfinite(L_ex) and L_ex > 0) else np.nan
            dm = dm_hln(-vals, np.zeros_like(vals))
            alpha = C / L if (L and np.isfinite(L) and L > 0) else np.nan
            rows.append({
                "test_year": ty, "shift_cost": sc_cost, "threshold": best_t,
                "days": len(te), "days_acted": int(sum(
                    1 for f in te if best_t <= 1.0 and np.sort(f[2])[::-1][N_HOURS-1] >= best_t)),
                "hours_changed": changed,
                "annual_value_per_MW": float(vals.sum()),
                "mean_daily": float(vals.mean()), "sd_daily": float(vals.std(ddof=1)),
                "sharpe": float(vals.mean() / vals.std(ddof=1)) if vals.std(ddof=1) else np.nan,
                "C": C, "L": L, "alpha": alpha,
                "pistar": alpha / (1 + alpha) if np.isfinite(alpha) else np.nan,
                "alpha_exante": alpha_ex,
                "pistar_exante": alpha_ex / (1 + alpha_ex) if np.isfinite(alpha_ex) else np.nan,
                "n_favourable": n_gain, "n_adverse": n_cost,
                "precision": n_gain / (n_gain + n_cost) if (n_gain + n_cost) else np.nan,
                "DM": dm["DM"], "p_value": dm["p_value"]})
            print(f"  {ty} cost {sc_cost:4.0f}: thr {best_t:.2f}  value {vals.sum():+9.0f}  "
                  f"alpha {alpha:.2f}  pi {rows[-1]['precision']:.3f} vs pi* "
                  f"{rows[-1]['pistar']:.3f}", flush=True)

    res = pd.DataFrame(rows)
    out = PROJECT_ROOT / "outputs" / "tables"
    res.to_csv(out / "load_shifting.csv", index=False)

    lines = ["# Load shifting with an energy balance\n",
             f"A load runs a {N_HOURS}-hour process every day and fixes the schedule before gate",
             "closure. Exactly N hours are used each day, so energy is conserved and every hour",
             "brought forward displaces one given up. The baseline is the climatologically",
             "cheapest N hours, estimated on training data only.\n",
             res.round(3).to_markdown(index=False), "",
             "\n## Cost and loss, measured over the substitutions actually made\n",
             res.groupby("shift_cost")[["C", "L", "alpha", "pistar", "precision"]]
                .mean().round(3).to_markdown()]
    text = "\n".join(lines)
    (PROJECT_ROOT / "experiments" / "14_load_shifting" / "FINDINGS.md").write_text(
        text, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("\n" + text)


if __name__ == "__main__":
    sys.exit(main())

