"""Does feed-in during negative hours fall after the February 2025 amendment?

Section 8.2 predicts that removing the run-length condition entirely -- so that every negative
hour suspends the market premium -- should push feed-in during negative hours toward zero for
directly marketed plants, leaving a residue from fallback remuneration, sub-threshold capacity
and must-run units. The amendment took effect on 25 February 2025 and the sample runs to the end
of that year, so the prediction is testable on data already in hand rather than on a future
vintage, as an earlier version of Section 8.2 assumed.

The outcome is the shortfall of realised renewable generation against the published day-ahead
forecast:

    shortfall_t = forecast_renewable_t - realised_renewable_t

The forecast is issued before the price is known, so it is a counterfactual for what would have
been fed in absent a price response, and a withdrawal of feed-in raises the shortfall.

Choosing the treated group is the whole difficulty. Comparing negative with non-negative hours
does not identify the amendment, because the amendment changes which negative hours are penalised
and so changes the composition of the negative-hour set: before it, only hours inside runs of at
least four were penalised, and those are the deepest-oversupply hours; after it, every negative
hour is, including mild one- and two-hour events that carry less curtailment. That comparison is
reported first because it is the obvious one and it is confounded, and the confounding runs in
the direction that would manufacture the wrong sign.

The identifying comparison holds the event fixed instead. Hours inside negative runs of one to
three hours were **not** penalised before 25 February 2025 and **are** penalised after: they are
the treated group. Hours inside runs of at least four were penalised in both regimes and are the
control. A behavioural response appears as a rise in the treated group's shortfall relative to
the control's.

What the test cannot do. The data do not identify plants by remuneration class, so the estimate
pools directly marketed plants with everything else and is a lower bound on the response of the
plants the rule actually binds. Nor can it separate a price response from a technical one:
negative hours are windy and sunny hours, in which grid constraints also bind. Both are stated in
Section 8.2 rather than left implicit.

Output: outputs/tables/amendment_test.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.features import load_hourly  # noqa: E402

OUT = PROJECT_ROOT / "outputs" / "tables"
AMEND = pd.Timestamp("2025-02-25", tz="UTC")

d = load_hourly()
act = pd.read_parquet(PROJECT_ROOT / "data" / "raw" / "actuals_de.parquet")
act = act[["Wind onshore", "Wind offshore", "Solar"]].resample("1h").mean()
act = act.reindex(d.index)

fc = d[["wind_onshore_fc", "wind_offshore_fc", "solar_fc"]].sum(axis=1)
re = act.sum(axis=1)
df = pd.DataFrame({"price": d["price"], "fc": fc, "re": re}).dropna()
df["shortfall"] = df.fc - df.re
df["neg"] = df.price < 0
df["post"] = df.index >= AMEND

# the comparison is confined to 2024-2025, so that the pre period is the closest available
# regime rather than the whole sample
w = df[df.index >= pd.Timestamp("2024-01-01", tz="UTC")]

rows = []
for post in (False, True):
    for neg in (False, True):
        s = w[(w.post == post) & (w.neg == neg)].shortfall
        rows.append(dict(period="post-amendment" if post else "pre-amendment",
                         hours="negative" if neg else "non-negative",
                         n=len(s), mean_shortfall_MW=s.mean(), sd=s.std()))
t = pd.DataFrame(rows)
print(t.round(1).to_string(index=False))

g = {(r["period"], r["hours"]): r["mean_shortfall_MW"] for _, r in t.iterrows()}
pre = g[("pre-amendment", "negative")] - g[("pre-amendment", "non-negative")]
post = g[("post-amendment", "negative")] - g[("post-amendment", "non-negative")]
did = post - pre
print(f"\nnegative minus non-negative shortfall:")
print(f"  before 25 Feb 2025 : {pre:+8.0f} MW")
print(f"  after              : {post:+8.0f} MW")
print(f"  difference in differences: {did:+.0f} MW")

def did(frame, treat_col, label):
    """Interaction regression with standard errors clustered by day."""
    f = frame.assign(post_i=frame.post.astype(float), t_i=frame[treat_col].astype(float))
    f["inter"] = f.post_i * f.t_i
    X = np.column_stack([np.ones(len(f)), f.post_i, f.t_i, f.inter])
    y = f.shortfall.to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    XtX_inv = np.linalg.inv(X.T @ X)
    meat = np.zeros((4, 4))
    for _, idx in pd.Series(np.arange(len(f)), index=f.index.normalize()).groupby(level=0):
        Xi, ri = X[idx.to_numpy()], resid[idx.to_numpy()]
        s = Xi.T @ ri
        meat += np.outer(s, s)
    se = np.sqrt(np.diag(XtX_inv @ meat @ XtX_inv))
    names = ["intercept", "post", "treated", "post x treated"]
    print(f"\n{label}: OLS, shortfall in MW, day-clustered standard errors, n = {len(f):,}")
    for n_, b_, s_ in zip(names, beta, se):
        print(f"  {n_:16} {b_:+9.1f}  ({s_:.1f})   t = {b_ / s_:+.2f}")
    return pd.DataFrame([dict(comparison=label, term=n_, coef=b_, se=s_, t=b_ / s_)
                         for n_, b_, s_ in zip(names, beta, se)])


r1 = did(w, "neg", "confounded: negative against non-negative hours")

# the identifying comparison: hours inside short negative runs, which the amendment newly
# penalises, against hours inside runs of at least four, which were penalised throughout
f = w.copy()
runs = f.neg.groupby((f.neg != f.neg.shift()).cumsum()).transform("sum").where(f.neg, 0)
f["run_len"] = runs
sub = f[f.neg]
sub = sub.assign(short_run=sub.run_len.between(1, 3))
print(f"\ntreated (runs of 1-3 h): {int(sub.short_run.sum()):,} hours; "
      f"control (runs >= 4 h): {int((~sub.short_run).sum()):,} hours")
for post in (False, True):
    for s_ in (True, False):
        m = sub[(sub.post == post) & (sub.short_run == s_)].shortfall
        print(f"  {'post' if post else 'pre ':4} {'1-3 h' if s_ else '>=4 h':6} "
              f"n = {len(m):5,}  mean shortfall {m.mean():7.0f} MW")
r2 = did(sub, "short_run", "identifying: newly penalised short runs against always-penalised")

t.to_csv(OUT / "amendment_test.csv", index=False)
pd.concat([r1, r2], ignore_index=True).to_csv(OUT / "amendment_test_regression.csv", index=False)
print("\nwrote amendment_test.csv and amendment_test_regression.csv")
