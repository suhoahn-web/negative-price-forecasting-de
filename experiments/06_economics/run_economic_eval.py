"""Economic evaluation at the EEG negative-price threshold.

Decision problem. Under the 2021 EEG rule, the sliding market premium for a renewable
generator is suspended for every hour of a block in which the day-ahead price is negative
for >= 4 consecutive hours, and the suspension applies FROM THE FIRST HOUR of the block.
A generator that can curtail therefore wants to know, before day-ahead gate closure,
whether the coming hours belong to such a block.

Per-hour economics for 1 MW of capacity, given the day-ahead price p and the reference
premium level (market value + premium) v:
    run in a suspended hour : revenue = p        (negative price paid, no premium)
    curtail                 : revenue = 0        (forgo generation, avoid the negative price)
    run in a normal hour    : revenue = max(p, ...) + premium -> approximated by v
Only hours where the plant would actually generate matter, so we weight by a normalized
solar/wind availability profile from the day-ahead forecast.

We compare decision rules, all using ONLY information available at gate closure:
  R0 never curtail                       (do nothing)
  R1 blind rule: curtail if forecast price < 0 -- but the price forecast is NOT available
                 ex ante, so we use the operational proxy: curtail if the hour is predicted
                 negative by the best price-free model
  R2 threshold rule on P(negative hour)
  R3 threshold rule on P(hour inside a >= 4 h block)   <- the EEG-aware rule
  RP perfect foresight                   (upper bound)
Thresholds for R2/R3 are chosen on the validation year, never on the test year.

Outputs: outputs/tables/economic_eval.csv, experiments/06_economics/ECONOMIC_NOTES.md
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.features import (FEATURES_AR, FEATURES_CAL, FEATURES_FC,  # noqa: E402
                          build, load_hourly)

TEST_YEARS = [2021, 2022, 2023, 2024, 2025]
SEED = 42
PREMIUM_REF = 60.0  # EUR/MWh reference remuneration level (sensitivity tested below)
FEATS = FEATURES_CAL + FEATURES_AR + FEATURES_FC


def fit_predict(d, target, train_mask, pred_mask):
    X = d[FEATS]
    ok = X.notna().all(axis=1).to_numpy()
    tr = train_mask & ok
    pr = pred_mask & ok
    m = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                       subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                       verbose=-1, n_jobs=4)
    m.fit(X[tr].to_numpy(), d[target].to_numpy()[tr])
    out = np.full(len(d), np.nan)
    out[pr] = m.predict_proba(X[pr].to_numpy())[:, 1]
    return out


def revenue(curtail: np.ndarray, price: np.ndarray, suspended: np.ndarray,
            weight: np.ndarray, premium: float) -> float:
    """EUR per MW of capacity over the evaluated hours."""
    run = ~curtail
    # suspended hour: revenue is the (negative) spot price; normal hour: premium level
    per_hour = np.where(suspended, price, premium)
    return float(np.sum(run * weight * per_hour))


def choose_threshold(p, price, suspended, weight, premium, grid=None) -> float:
    """Tune on validation revenue. The grid INCLUDES 1.01 = 'never act', so a rational
    operator is allowed to decline to use the forecast when it does not pay."""
    grid = grid if grid is not None else np.append(np.linspace(0.05, 0.95, 19), 1.01)
    best_t, best_v = 1.01, -np.inf
    for t in grid:
        v = revenue(p >= t, price, suspended, weight, premium)
        if v > best_v:
            best_t, best_v = float(t), v
    return best_t


def flex_value(consume: np.ndarray, price: np.ndarray, weight: np.ndarray,
               shift_cost: float) -> float:
    """Flexible load / storage: value of shifting consumption INTO an hour.

    Consuming 1 MWh at price p costs p (a gain when p < 0). Shifting has a cost
    (efficiency loss / opportunity cost) charged whenever the flexible block is used.
    """
    return float(np.sum(consume * weight * (-price - shift_cost)))


def choose_flex_threshold(p, price, weight, shift_cost, grid=None) -> float:
    grid = grid if grid is not None else np.append(np.linspace(0.05, 0.95, 19), 1.01)
    best_t, best_v = 1.01, -np.inf
    for t in grid:
        v = flex_value(p >= t, price, weight, shift_cost)
        if v > best_v:
            best_t, best_v = float(t), v
    return best_t


def main() -> None:
    d, _ = build(load_hourly())
    d["avail"] = (d["res_fc"] / d["res_fc"].max()).clip(0, 1)  # generation weight
    suspended_all = d["y_run4"].to_numpy().astype(bool)
    price_all = d["price"].to_numpy()
    years = d.index.year

    rows = []
    for ty in TEST_YEARS:
        tr = years < ty - 1
        va = years == ty - 1
        te = years == ty
        if tr.sum() < 2000 or va.sum() < 500 or te.sum() < 500:
            continue

        p_neg = fit_predict(d, "y_neg", tr | va, va | te)
        p_run = fit_predict(d, "y_run4", tr | va, va | te)

        w = d["avail"].to_numpy()

        # --- Use case B: flexible load / storage (opposite asymmetry) ---
        # Being wrong costs only the shift cost; being right earns |negative price|.
        wf = np.ones(len(d))  # a flexible block is available in any hour
        for shift_cost in (0.0, 5.0, 15.0):
            t_fneg = choose_flex_threshold(p_neg[va], price_all[va], wf[va], shift_cost)
            t_frun = choose_flex_threshold(p_run[va], price_all[va], wf[va], shift_cost)
            naive = flex_value(np.zeros(te.sum(), bool), price_all[te], wf[te], shift_cost)
            perfect_f = flex_value(price_all[te] < 0, price_all[te], wf[te], shift_cost)
            v_neg = flex_value(p_neg[te] >= t_fneg, price_all[te], wf[te], shift_cost)
            v_run = flex_value(p_run[te] >= t_frun, price_all[te], wf[te], shift_cost)
            rows.append({"use_case": "flexible_load", "test_year": ty,
                         "param": shift_cost, "n_hours": int(te.sum()),
                         "do_nothing": naive, "pred_pneg": v_neg, "pred_prun4": v_run,
                         "perfect": perfect_f, "thr_pneg": t_fneg, "thr_prun4": t_frun,
                         "gain_vs_nothing": max(v_neg, v_run) - naive,
                         "captured_of_perfect": ((max(v_neg, v_run) - naive) /
                                                 (perfect_f - naive)) if perfect_f > naive else np.nan})

        # --- Use case A: premium-eligible generator curtailment (EEG) ---
        for premium in (40.0, PREMIUM_REF, 90.0):
            args_va = (price_all[va], suspended_all[va], w[va], premium)
            args_te = (price_all[te], suspended_all[te], w[te], premium)

            t2 = choose_threshold(p_neg[va], *args_va)
            t3 = choose_threshold(p_run[va], *args_va)

            base = revenue(np.zeros(te.sum(), bool), *args_te)
            perfect = revenue(suspended_all[te], *args_te)
            r1 = revenue(p_neg[te] >= 0.5, *args_te)
            r2 = revenue(p_neg[te] >= t2, *args_te)
            r3 = revenue(p_run[te] >= t3, *args_te)

            n_hours = int(te.sum())
            rows.append({
                "use_case": "generator_curtailment",
                "test_year": ty, "premium": premium, "n_hours": n_hours,
                "R0_never_curtail": base, "R1_blind_p50": r1,
                "R2_tuned_pneg": r2, "R3_tuned_prun4": r3, "RP_perfect": perfect,
                "thr_R2": t2, "thr_R3": t3,
                "gain_R3_vs_R0": r3 - base, "gain_R3_vs_R1": r3 - r1,
                "captured_of_perfect": (r3 - base) / (perfect - base) if perfect > base else np.nan,
            })
        print(f"  {ty} done", flush=True)

    res = pd.DataFrame(rows)
    out = PROJECT_ROOT / "outputs" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "economic_eval.csv", index=False)

    lines = ["# Economic Evaluation\n",
             "EUR per MW per test year. Thresholds tuned on the preceding year only, and the",
             "tuning grid includes 'never act', so a rational operator may decline to use the",
             "forecast when it does not pay.\n",
             "\n## Use case B — flexible load / storage (value of consuming in negative hours)\n",
             "| year | shift cost | do nothing | P(neg) rule | P(run>=4h) rule | perfect | "
             "gain | % of perfect |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    fl = res[res.use_case == "flexible_load"]
    for _, r in fl.sort_values(["param", "test_year"]).iterrows():
        pct = (f"{100 * r.captured_of_perfect:.0f}%"
               if pd.notna(r.captured_of_perfect) else "n/a")
        lines.append(f"| {int(r.test_year)} | {r.param:.0f} | {r.do_nothing:,.0f} | "
                     f"{r.pred_pneg:,.0f} | {r.pred_prun4:,.0f} | {r.perfect:,.0f} | "
                     f"{r.gain_vs_nothing:+,.0f} | {pct} |")

    lines.append("\n\n## Use case A — premium-eligible generator curtailment (EEG rule)\n")
    gen = res[res.use_case == "generator_curtailment"]
    for premium in sorted(gen.premium.dropna().unique()):
        sub = gen[gen.premium == premium]
        lines.append(f"\n## Reference level {premium:.0f} EUR/MWh\n")
        lines.append("| year | never curtail | blind p>=0.5 | tuned P(neg) | "
                     "**tuned P(run>=4h)** | perfect | gain vs never | % of perfect |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in sub.iterrows():
            lines.append(
                f"| {int(r.test_year)} | {r.R0_never_curtail:,.0f} | {r.R1_blind_p50:,.0f} | "
                f"{r.R2_tuned_pneg:,.0f} | **{r.R3_tuned_prun4:,.0f}** | {r.RP_perfect:,.0f} | "
                f"{r.gain_R3_vs_R0:+,.0f} | {100*r.captured_of_perfect:.0f}% |")
    # --- the payoff asymmetry that explains both use cases ---
    neg_mask = price_all < 0
    mean_neg = float(np.abs(price_all[neg_mask]).mean())
    med_neg = float(np.abs(np.median(price_all[neg_mask])))
    lines.append("\n\n## Why the two use cases differ: the payoff asymmetry\n")
    lines.append(f"- mean magnitude of a negative price: {mean_neg:.2f} EUR/MWh "
                 f"(median {med_neg:.2f})")
    lines.append("| side of the market | gain when correct | cost of a false alarm | ratio |")
    lines.append("|---|---:|---:|---:|")
    for premium in (40.0, 60.0, 90.0):
        lines.append(f"| premium generator (curtail), ref {premium:.0f} | {mean_neg:.2f} | "
                     f"{premium:.2f} | 1 : {premium / mean_neg:.0f} |")
    for sc in (0.0, 5.0, 15.0):
        lines.append(f"| flexible load (consume), shift cost {sc:.0f} | "
                     f"{mean_neg + 0:.2f} | {sc:.2f} | "
                     + (f"1 : {sc / mean_neg:.1f} |" if sc > 0 else "no downside |"))
    lines.append("\nA premium-eligible generator must be right roughly "
                 f"{100 * (1 / (1 + mean_neg / 60)):.0f}% of the time before curtailment on a "
                 "forecast breaks even at a 60 EUR/MWh reference level, because a false alarm "
                 "forfeits the full remuneration while a correct call saves only the small "
                 "negative price. The flexible-load side faces the opposite asymmetry, which "
                 "is why the SAME forecasts are profitable there.")

    text = "\n".join(lines)
    (PROJECT_ROOT / "experiments" / "06_economics").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "experiments" / "06_economics" / "ECONOMIC_NOTES.md").write_text(
        text, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(text)


if __name__ == "__main__":
    sys.exit(main())
