"""Formal test of duration dependence in negative-price episodes.

WHY
Section 4.1 previously asserted that the conditional continuation probability "falls
monotonically" with episode age. That is false: it rises from 0.811 at age 3 to 0.828 at age 4.
The direction of the tendency may still be real, but an eyeballed monotonicity claim is not
evidence for it, so this script tests it properly.

DESIGN
Build an episode-hour panel. Each episode of length L contributes one row for every age
k = 1 .. L, with an exit indicator that is 1 in its final hour. The conditional probability of
exiting at age k given survival to k is then a binary outcome on that panel, and duration
dependence is the question of whether it varies with k.

  primary        logit(P(exit at age k | survived to k)) = b0 + b1 * k
  robustness     complementary log-log link; age as a categorical factor; year fixed effects;
                 a 2022 indicator for the gas-crisis regime

Standard errors are clustered by episode, because the ages within an episode are not
independent. A positive b1 means the exit probability rises with age, which is POSITIVE
duration dependence in the standard terminology, and corresponds to the continuation
probability falling.

Outputs
  outputs/tables/duration_dependence.csv
  experiments/15_duration_test/FINDINGS.md
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.features import build, load_hourly  # noqa: E402

SEED = 42


def episode_panel(d: pd.DataFrame) -> pd.DataFrame:
    neg = d["price"] < 0
    grp = (neg != neg.shift()).cumsum()
    lens = neg.groupby(grp).sum()
    starts = d.index.to_series().groupby(grp).first()
    rows = []
    for gid, L in lens[lens > 0].items():
        t0 = starts[gid]
        for k in range(1, int(L) + 1):
            rows.append({"episode": int(gid), "age": k, "exit": int(k == L),
                         "year": t0.year, "start_hour": t0.hour,
                         "gas_crisis": int(t0.year == 2022)})
    return pd.DataFrame(rows)


def fit(panel, formula_cols, label, rows):
    X = sm.add_constant(panel[formula_cols].astype(float), has_constant="add")
    y = panel["exit"].to_numpy()
    fam = sm.families.Binomial()
    m = sm.GLM(y, X, family=fam).fit(cov_type="cluster",
                                     cov_kwds={"groups": panel["episode"]})
    for name in formula_cols:
        rows.append({"model": label, "term": name,
                     "coef": m.params[name], "se": m.bse[name],
                     "z": m.tvalues[name], "p": m.pvalues[name],
                     "ci_lo": m.conf_int().loc[name, 0],
                     "ci_hi": m.conf_int().loc[name, 1],
                     "n_obs": int(m.nobs),
                     "n_episodes": panel["episode"].nunique()})
    return m


def main() -> None:
    d, _ = build(load_hourly())
    panel = episode_panel(d)
    print(f"panel: {len(panel):,} episode-hours from {panel.episode.nunique()} episodes",
          flush=True)

    rows = []
    m1 = fit(panel, ["age"], "logit, age linear", rows)
    print(f"  logit age coef {m1.params['age']:+.4f} (p={m1.pvalues['age']:.3g})", flush=True)

    # cloglog
    X = sm.add_constant(panel[["age"]].astype(float), has_constant="add")
    m2 = sm.GLM(panel["exit"].to_numpy(), X,
                family=sm.families.Binomial(sm.families.links.CLogLog())).fit(
        cov_type="cluster", cov_kwds={"groups": panel["episode"]})
    rows.append({"model": "cloglog, age linear", "term": "age", "coef": m2.params["age"],
                 "se": m2.bse["age"], "z": m2.tvalues["age"], "p": m2.pvalues["age"],
                 "ci_lo": m2.conf_int().loc["age", 0], "ci_hi": m2.conf_int().loc["age", 1],
                 "n_obs": int(m2.nobs), "n_episodes": panel.episode.nunique()})

    # with a gas-crisis indicator and year effects
    p2 = panel.copy()
    for y in sorted(p2.year.unique())[1:]:
        p2[f"y{y}"] = (p2.year == y).astype(int)
    ycols = [c for c in p2.columns if c.startswith("y2")]
    fit(p2, ["age"] + ycols, "logit, age + year effects", rows)
    fit(panel, ["age", "gas_crisis"], "logit, age + 2022 indicator", rows)

    # age as a categorical factor, to see the shape without imposing linearity
    p3 = panel[panel.age <= 8].copy()
    for k in range(2, 9):
        p3[f"age{k}"] = (p3.age == k).astype(int)
    fit(p3, [f"age{k}" for k in range(2, 9)], "logit, age categorical (k<=8)", rows)

    res = pd.DataFrame(rows)
    out = PROJECT_ROOT / "outputs" / "tables"
    res.to_csv(out / "duration_dependence.csv", index=False)

    a = res[(res.model == "logit, age linear") & (res.term == "age")].iloc[0]
    direction = ("positive duration dependence: the exit probability RISES with age, so the "
                 "continuation probability falls") if a.coef > 0 else (
                 "negative duration dependence: the exit probability FALLS with age")
    verdict = "significant at 5%" if a.p < 0.05 else "not significant at 5%"

    lines = ["# Duration dependence, formally tested\n",
             f"Episode-hour panel: {len(panel):,} rows from {panel.episode.nunique()} episodes.",
             "Standard errors clustered by episode.\n",
             res.round(4).to_markdown(index=False), "",
             f"\n**Primary result.** In the linear-age logit the coefficient on age is "
             f"{a.coef:+.4f} (cluster-robust SE {a.se:.4f}, p = {a.p:.3g}, 95% CI "
             f"[{a.ci_lo:+.4f}, {a.ci_hi:+.4f}]). This is {verdict}.",
             f"\nInterpretation: {direction}.",
             "\n**Terminology.** In standard survival terminology the hazard is the conditional",
             "probability of ENDING. The quantity tabulated in Section 4.1 is its complement, the",
             "conditional continuation probability. The manuscript now uses the latter name."]
    text = "\n".join(lines)
    (PROJECT_ROOT / "experiments" / "15_duration_test" / "FINDINGS.md").write_text(
        text, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("\n" + text)


if __name__ == "__main__":
    sys.exit(main())
