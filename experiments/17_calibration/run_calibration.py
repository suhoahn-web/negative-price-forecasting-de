"""Gate 3: is the probability calibrated, and does Section 7 depend on it?

Sections 6 evaluates ranking and a proper score. Section 7 then converts the probability into a
decision by comparing it with a tuned threshold, and a threshold rule inherits whatever
miscalibration the probability carries. Reporting average precision alone leaves that unexamined,
so this script reports the three diagnostics a probabilistic forecast is normally held to.

  reliability curve       observed frequency against predicted probability, in equal-count bins
  calibration intercept   from the Cox recalibration regression of the outcome on the forecast
  and slope               logit. Perfect calibration is intercept 0 and slope 1; a slope below 1
                          means the forecasts are too extreme, above 1 too timid
  Murphy decomposition    Brier = reliability - resolution + uncertainty, so that the part of the
                          score attributable to miscalibration is separated from the part
                          attributable to the forecast's ability to discriminate

The decision-relevant question is narrower than "is it calibrated". Section 7 tunes its acting
threshold on the preceding year's out-of-sample predictions, so a monotone distortion is absorbed
by the threshold and cannot change the decision. What would change it is a distortion that moves
between the tuning year and the test year, which is what the year-by-year slopes below measure.

Outputs
  outputs/tables/calibration.csv
  outputs/figures/fig4_calibration.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRED = PROJECT_ROOT / "outputs" / "preds" / "rolling730"
OUT_T = PROJECT_ROOT / "outputs" / "tables"
OUT_F = PROJECT_ROOT / "outputs" / "figures"
OUT_F.mkdir(parents=True, exist_ok=True)

COL = "lgbm|cal+AR(D-1)+FC"          # the primary model on the full feature set
TASKS = ["y_neg", "y_run4", "y_day4"]
YEARS = [2021, 2022, 2023, 2024, 2025]
EPS = 1e-6
NBINS = 10


def cox(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Cox recalibration: regress the outcome on the forecast logit.

    Intercept 0 and slope 1 is perfect calibration. The slope is the diagnostic that matters for
    a threshold rule, because a slope far from 1 means the ordering of probabilities relative to
    any fixed cut-off is distorted.
    """
    z = np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS))).reshape(-1, 1)
    m = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000).fit(z, y)
    return float(m.intercept_[0]), float(m.coef_[0][0])


def murphy(y: np.ndarray, p: np.ndarray, nbins: int = NBINS) -> dict:
    """Brier = reliability - resolution + uncertainty, on equal-count bins."""
    order = np.argsort(p)
    bins = np.array_split(order, nbins)
    base = y.mean()
    rel = res = 0.0
    for b in bins:
        if not len(b):
            continue
        w = len(b) / len(y)
        rel += w * (p[b].mean() - y[b].mean()) ** 2
        res += w * (y[b].mean() - base) ** 2
    unc = base * (1 - base)
    return dict(reliability=rel, resolution=res, uncertainty=unc,
                brier=float(np.mean((p - y) ** 2)), bss=1 - np.mean((p - y) ** 2) / unc)


def recalibrate(p_fit, y_fit, p_apply):
    """Apply a Cox recalibration estimated on one year to the next.

    This is the operational version of the diagnostic: a practitioner who needs a calibrated
    probability rather than a ranking fits the intercept and slope on the year before the one
    being forecast, which is the same data the acting threshold of Section 7 already uses, so no
    new assumption enters.
    """
    z = np.log(np.clip(p_fit, EPS, 1 - EPS) / (1 - np.clip(p_fit, EPS, 1 - EPS))).reshape(-1, 1)
    m = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000).fit(z, y_fit)
    za = np.log(np.clip(p_apply, EPS, 1 - EPS)
                / (1 - np.clip(p_apply, EPS, 1 - EPS))).reshape(-1, 1)
    return m.predict_proba(za)[:, 1]


rows, curves = [], {}
for task in TASKS:
    for year in YEARS:
        f = PRED / f"{task}_{year}.parquet"
        if not f.exists():
            print(f"  missing {f.name}")
            continue
        d = pd.read_parquet(f)
        y, p = d["y"].to_numpy().astype(float), d[COL].to_numpy()
        a, b = cox(y, p)
        m = murphy(y, p)
        # The Cox slope over the whole sample is dominated by the mass of near-zero forecasts,
        # where the logit is large and negative and no decision is ever taken. Section 7 acts
        # only on high probabilities, so the slope is also computed on the decision region: the
        # observations whose forecast exceeds the base rate, which is where any acting threshold
        # must lie.
        sel = p > y.mean()
        a_d, b_d = (cox(y[sel], p[sel]) if sel.sum() > 30 and 0 < y[sel].mean() < 1
                    else (np.nan, np.nan))
        top = p >= np.quantile(p, 0.99)

        # out-of-sample recalibration: fit on the preceding test year, apply to this one
        prev = PRED / f"{task}_{year - 1}.parquet"
        rc_brier = rc_slope = np.nan
        if prev.exists():
            dp = pd.read_parquet(prev)
            p_rc = recalibrate(dp[COL].to_numpy(), dp["y"].to_numpy().astype(float), p)
            rc_brier = float(np.mean((p_rc - y) ** 2))
            rc_slope = cox(y, p_rc)[1]

        rows.append(dict(task=task, test_year=year, n=len(y), base_rate=y.mean(),
                         brier_recal=rc_brier, slope_recal=rc_slope,
                         cal_intercept=a, cal_slope=b,
                         cal_intercept_decision=a_d, cal_slope_decision=b_d,
                         n_decision=int(sel.sum()),
                         mean_p_top1=float(p[top].mean()), freq_top1=float(y[top].mean()),
                         **m))
        order = np.argsort(p)
        curves[(task, year)] = [(p[k].mean(), y[k].mean(), len(k))
                                for k in np.array_split(order, NBINS)]

t = pd.DataFrame(rows)
t.to_csv(OUT_T / "calibration.csv", index=False)
print(t.round(4).to_string(index=False))
print(f"\ncalibration slope, whole sample: mean {t.cal_slope.mean():.3f}, "
      f"range {t.cal_slope.min():.3f} to {t.cal_slope.max():.3f}")
d_ = t.cal_slope_decision.dropna()
print(f"calibration slope, decision region: mean {d_.mean():.3f}, "
      f"range {d_.min():.3f} to {d_.max():.3f}")
print(f"top 1% of forecasts: mean predicted {t.mean_p_top1.mean():.3f} against observed "
      f"{t.freq_top1.mean():.3f}")
print(f"reliability is {t.reliability.mean() / t.brier.mean() * 100:.1f}% of the Brier score "
      f"on average; resolution is {t.resolution.mean() / t.uncertainty.mean() * 100:.1f}% "
      f"of uncertainty")
rc = t.dropna(subset=["brier_recal"])
print(f"\nout-of-sample recalibration on the preceding year, "
      f"{len(rc)} of {len(t)} combinations")
print(f"  Brier: {rc.brier.mean():.4f} raw -> {rc.brier_recal.mean():.4f} recalibrated")
print(f"  improves in {int((rc.brier_recal < rc.brier).sum())} of {len(rc)}")
print(f"  slope: {rc.cal_slope.mean():.2f} raw -> {rc.slope_recal.mean():.2f} recalibrated")
print(f"Brier skill score: mean {t.bss.mean():.3f}, "
      f"range {t.bss.min():.3f} to {t.bss.max():.3f}")

# year-to-year drift in the slope is what a tuned threshold cannot absorb
for task in TASKS:
    s = t[t.task == task].sort_values("test_year").cal_slope.to_numpy()
    print(f"  {task:7} slope by year " + " ".join(f"{v:.2f}" for v in s)
          + f"   max year-on-year change {np.abs(np.diff(s)).max():.2f}")

# ------------------------------------------------------------------ figure
plt.rcParams.update({"font.size": 8, "font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.dpi": 600, "savefig.bbox": "tight"})
GREY, DARK = "0.72", "0.15"
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
for ax, task in zip(axes, TASKS):
    ax.plot([0, 1], [0, 1], color=GREY, lw=.8, ls="--", zorder=1)
    for i, year in enumerate(YEARS):
        c = curves.get((task, year))
        if not c:
            continue
        xs = [x for x, _, _ in c]
        ys = [v for _, v, _ in c]
        ax.plot(xs, ys, "o-", ms=2.2, lw=.9, color=str(0.62 - 0.12 * i), label=str(year))
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("forecast probability")
    ax.set_title(f"({'abc'[TASKS.index(task)]})", fontsize=8, loc="left")
axes[0].set_ylabel("observed frequency")
axes[2].legend(frameon=False, fontsize=6, loc="upper left")
fig.tight_layout()
fig.savefig(OUT_F / "fig4_calibration.png")
print(f"\nwrote {OUT_T / 'calibration.csv'} and fig4_calibration.png")

