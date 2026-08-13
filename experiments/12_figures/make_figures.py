"""Manuscript figures.

Fig 1  day-ahead trading timeline and the information set (schematic, after Kath & Ziel 2018)
Fig 2  negative-price episodes: duration distribution, survival, and the threshold artefact
Fig 3  relative economic value against the cost-loss ratio (the Murphy 1977 value curve)

All figures are greyscale-safe and sized for a single journal column unless noted.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.features import build, load_hourly, run_lengths  # noqa: E402

OUT = PROJECT_ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.bbox": "tight"})
GREY, DARK, MID = "0.75", "0.15", "0.45"


# --------------------------------------------------------------------------- Fig 1
def fig1_timeline():
    fig, ax = plt.subplots(figsize=(6.5, 2.1))
    ax.axhline(0, color=DARK, lw=1)
    events = [(-24, "D-2 12:00\nauction for D-1 clears", "in"),
              (-13, "D-1 ~10:00\nday-ahead RES and\nload forecasts for D", "in"),
              (0, "D-1 12:00\nGATE CLOSURE\nfor day D", "gate"),
              (0.7, "D-1 12:42\nprices for D published", "out"),
              (14, "day D\ndelivery, realised\ngeneration known", "out")]
    for x, label, kind in events:
        col = DARK if kind != "out" else MID
        ax.plot([x], [0], "o", color=col, ms=6 if kind == "gate" else 4, zorder=3)
        va, y = ("bottom", 0.13) if kind != "out" else ("top", -0.13)
        ax.annotate(label, (x, y), ha="center", va=va, fontsize=6.5, color=col)
    ax.axvspan(-30, 0, color=GREY, alpha=.45, lw=0)
    ax.annotate("information set: everything left of the gate", (-15, -0.42), ha="center",
                fontsize=7, color=DARK)
    ax.annotate("not available", (7.5, 0.42), ha="center", fontsize=7, color=MID)
    ax.axvline(0, color=DARK, lw=1.2, ls="--")
    ax.set_xlim(-30, 20); ax.set_ylim(-0.55, 0.55)
    ax.set_yticks([]); ax.set_xticks([])
    ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False)
    fig.savefig(OUT / "fig1_timeline.png")
    plt.close(fig)
    print("fig1 written")


# --------------------------------------------------------------------------- Fig 2
def episodes(flag: pd.Series) -> np.ndarray:
    f = flag.fillna(False).astype(bool)
    grp = (f != f.shift()).cumsum()
    lens = f.groupby(grp).sum()
    return lens[lens > 0].to_numpy()


def fig2_episodes(d: pd.DataFrame):
    neg = d["price"] < 0
    L = episodes(neg)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))

    ax = axes[0]
    bins = np.arange(0.5, 25.5, 1)
    ax.hist(np.clip(L, 0, 24), bins=bins, color=GREY, edgecolor=DARK, lw=.4)
    ax.axvline(4, color=DARK, ls="--", lw=1)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    ax.annotate("EEG 4 h trigger", (5.0, ax.get_ylim()[1] * .93), fontsize=6.5, color=DARK,
                ha="left", va="top")
    ax.set_xlabel("episode length (h)"); ax.set_ylabel("episodes")
    ax.set_title("(a) duration distribution", fontsize=8, loc="left")

    # (b) hazard, shown ONLY where the at-risk set supports it. Beyond that the estimate is
    # driven by a handful of episodes and swings between 0.5 and 1.0; plotting it would
    # misrepresent the precision of the flat-hazard claim rather than support it.
    MIN_AT_RISK = 20
    ax = axes[1]
    ks = np.arange(1, 25)
    at_risk = np.array([(L >= k).sum() for k in ks])
    surv = np.array([(L > k).sum() / n if n else np.nan for k, n in zip(ks, at_risk)])
    ok = at_risk >= MIN_AT_RISK
    ax.plot(ks[ok], surv[ok], "o-", color=DARK, ms=2.5, lw=1, label="empirical hazard")
    ax.plot(ks[~ok], surv[~ok], "o", color="0.8", ms=2, label=f"< {MIN_AT_RISK} at risk")
    p = 0.809
    ax.axhline(p, color=MID, ls=":", lw=1.2, label=f"geometric fit p={p:.3f}")
    kmax = ks[ok].max()
    ax.axvline(kmax + .5, color="0.8", lw=.8)
    ax.annotate(f"n at risk < {MIN_AT_RISK}", (kmax + 1.0, .12), fontsize=5.5, color="0.5",
                rotation=90, va="bottom")
    ax.set_ylim(0, 1.05); ax.set_xlabel("episode age k (h)")
    ax.set_ylabel("P(L > k | L $\\geq$ k)")
    ax.legend(fontsize=5.5, frameon=False, loc="lower left")
    ax.set_title("(b) continuation hazard is flat", fontsize=8, loc="left")
    print(f"  hazard shown to k={kmax} (at risk {at_risk[ok][-1]}); "
          f"range over that span {surv[ok].min():.2f}-{surv[ok].max():.2f}")

    # (c) mean duration against median survival. A definition that found more persistence would
    # move both; a fixed threshold moves only the mean, because it records a level shift as a
    # few enormous episodes. Reporting a fitted half-life here would import the geometric model
    # that panel (b) rejects.
    ax = axes[2]
    defs = {"price < 0": neg,
            "fixed > 200": d["price"] > 200,
            "rolling p95": d["price"] > d["price"].rolling(720, min_periods=200).quantile(.95),
            "rolling p99": d["price"] > d["price"].rolling(720, min_periods=200).quantile(.99)}
    means, meds, names = [], [], []
    for name, f in defs.items():
        e = episodes(f)
        if len(e) < 20:
            continue
        ks = np.arange(1, 49)
        S = np.array([(e > k).sum() / len(e) for k in ks])
        i = int(np.argmax(S <= 0.5))
        med = ks[i - 1] + (S[i - 1] - .5) / (S[i - 1] - S[i]) if i > 0 else 0.5
        names.append(name); means.append(e.mean()); meds.append(med)
    y = np.arange(len(names))[::-1]
    ax.barh(y + .18, means, height=.34, color=GREY, edgecolor=DARK, lw=.4, label="mean length")
    ax.barh(y - .18, meds, height=.34, color="0.35", edgecolor=DARK, lw=.4,
            label="median survival")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=6.5)
    for yi, m, md in zip(y, means, meds):
        ax.annotate(f"{m:.1f}", (m + .2, yi + .18), va="center", fontsize=5.5, color=DARK)
        ax.annotate(f"{md:.1f}", (md + .2, yi - .18), va="center", fontsize=5.5, color=DARK)
    ax.set_xlabel("hours")
    ax.legend(fontsize=5.5, frameon=False, loc="lower right")
    ax.set_title("(c) the threshold artefact", fontsize=8, loc="left")
    ax.set_xlim(0, max(means) * 1.3)

    fig.tight_layout()
    fig.savefig(OUT / "fig2_episodes.png")
    plt.close(fig)
    print("fig2 written; mean vs median survival:",
          {n: (round(m, 2), round(md, 2)) for n, m, md in zip(names, means, meds)})


# --------------------------------------------------------------------------- Fig 3
def _roc(task: str) -> tuple:
    """Pooled (H, F, base rate) over all decision thresholds, from saved test-year predictions."""
    preds = sorted((PROJECT_ROOT / "outputs" / "preds" / "monthly").glob(f"{task}_*.parquet"))
    if not preds:
        preds = sorted((PROJECT_ROOT / "outputs" / "preds").glob(f"{task}_*.parquet"))
    ys, ps = [], []
    for f in preds:
        df = pd.read_parquet(f)
        col = "lgbm|cal+AR(D-1)+FC"
        if col not in df.columns:
            continue
        m = df[col].notna()
        ys.append(df.loc[m, "y"].to_numpy())
        ps.append(df.loc[m, col].to_numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    order = np.argsort(-p)
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    P, N = y.sum(), (1 - y).sum()
    return tp / P, fp / N, float(P / (P + N))


def fig3_value_curve():
    be = pd.read_csv(PROJECT_ROOT / "outputs" / "tables" / "costloss_breakeven.csv")
    pooled = pd.read_csv(PROJECT_ROOT / "outputs" / "tables" / "costloss_pooled.csv")
    alphas = np.linspace(0.02, 8, 300)

    # Richardson (2000) / Zhu et al. (2002) potential-value envelope: at each cost-loss ratio,
    # the value obtainable at the BEST decision threshold. Drawing a single fixed operating
    # point instead would misstate the value available to participants whose tuned threshold
    # differs, which is exactly what Table 8 records.
    curves = {}
    for actor, task in (("flexible_load", "y_neg"), ("premium_generator", "y_run4")):
        H, F, s = _roc(task)
        V = []
        for a in alphas:
            eref = min(s, (1 - s) * a)
            e = (1 - H) * s + F * (1 - s) * a
            V.append(1 - e.min() / eref if eref > 0 else np.nan)
        curves[actor] = np.array(V)

    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    ax.plot(alphas, curves["flexible_load"], "-", color=DARK, lw=1.3,
            label="attainable value, hourly negative-price target")
    ax.plot(alphas, curves["premium_generator"], "--", color=MID, lw=1.3,
            label="attainable value, $\\geq$4 h run target")
    ax.axhline(0, color="0.6", lw=.8)

    # the load-shifting decision of Section 7.5, whose alpha is measured on realised
    # deviating days rather than assumed; plotted as the range across shift costs and years
    try:
        ls = pd.read_csv(PROJECT_ROOT / "outputs" / "tables" / "load_shifting.csv")
        ls = ls[(ls.days_acted > 0) & np.isfinite(ls.alpha)]
        if len(ls):
            lo, hi = ls.alpha.min(), ls.alpha.max()
            ax.axvspan(lo, hi, color=DARK, alpha=.10, lw=0)
            ax.annotate("load shifting\n(Section 7.5)", ((lo + hi) / 2, .62), ha="center",
                        fontsize=5.5, color="0.25")
    except FileNotFoundError:
        pass

    YLO = -0.5
    for _, r in pooled.iterrows():
        a = r["alpha"]
        if a <= 0:
            continue
        row = be[(be.actor == r["actor"]) & (be.param == r["param"])]
        H, F, s = row["H"].mean(), row["F"].mean(), row["base_rate"].mean()
        eref = min(s, (1 - s) * a)
        V = 1 - ((1 - H) * s + F * (1 - s) * a) / eref if eref > 0 else np.nan
        col = DARK if r["actor"] == "flexible_load" else MID
        lab = ("load, c=%g" % r["param"]) if r["actor"] == "flexible_load" \
            else ("gen, ref %g" % r["param"])
        if V < YLO:                              # off scale: mark at the edge and give the value
            ax.plot([a], [YLO], "v", color=col, ms=5, mec="white", mew=.6, clip_on=False,
                    zorder=5)
            ax.annotate(f"{lab}\nV = {V:.1f}", (a, YLO + .04), ha="center", va="bottom",
                        fontsize=5.2, color="0.25")
        else:
            ax.plot([a], [V], "o", color=col, ms=5, mec="white", mew=.7, zorder=5)
            ax.plot([a, a], [min(V, 0), max(V, 0)], ls=":", color=col, lw=.7, zorder=1)
            ax.annotate(lab, (a, V + (.045 if V >= 0 else -.045)), ha="center",
                        va="bottom" if V >= 0 else "top", fontsize=5.5, color="0.25")

    # a published decision-maker on the same axis: Christensen, Hurn & Lindsay (2012) weight a
    # missed spike three times as heavily as a false alarm, i.e. L/C = 3 and alpha = 1/3
    ax.axvline(1 / 3, color="0.5", lw=.8, ls="-.")
    ax.annotate("Christensen et al. (2012)\nretailer, $\\alpha=1/3$", (1 / 3 + .13, .97),
                fontsize=5.5, color="0.35", va="top")

    ax.set_xlabel(r"cost-loss ratio  $\alpha = C/L$")
    ax.set_ylabel("relative economic value $V$")
    ax.set_xlim(0, 8); ax.set_ylim(YLO, 1.0)
    ax.legend(fontsize=6, frameon=False, loc="upper right")
    fig.savefig(OUT / "fig3_value_curve.png")
    plt.close(fig)
    for k, v in curves.items():
        cross = alphas[np.argmax(v <= 0)] if (v <= 0).any() else np.nan
        print(f"fig3: {k} envelope crosses V=0 at alpha = {cross:.2f}")


if __name__ == "__main__":
    d, _ = build(load_hourly())
    fig1_timeline()
    fig2_episodes(d)
    fig3_value_curve()
    print("all figures in", OUT)
