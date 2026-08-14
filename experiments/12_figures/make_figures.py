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
# 600 dpi on save: Elsevier asks for 600 dpi for combination line/halftone art. The on-screen
# figure dpi is left low so the layout is composed at the printed size rather than scaled up.
# The guide names Arial, Courier, Times New Roman and Symbol as the fonts to aim for in
# illustrations; matplotlib's default DejaVu Sans is none of them.
plt.rcParams.update({"font.size": 8, "font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.dpi": 600, "savefig.bbox": "tight"})
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
    # bare panel identifiers only: the guide asks for no titles above the plots, since the
    # caption carries the description. The letters stay because the caption refers to them.
    ax.set_title("(a)", fontsize=8, loc="left")

    # (b) CONTINUATION PROBABILITY c(k) = P(L > k | L >= k). Two things this panel must not do.
    # It must not call c(k) a hazard: in survival terminology the hazard is the probability of
    # ENDING, which is 1 - c(k). And it must not draw the geometric fit as a description of the
    # data, because Section 4.1 rejects that model (chi2 = 72.7 on 10 df). The earlier version of
    # this figure did both, and titled the panel with a claim the paper's own test refuses.
    #
    # The plotted range matches Table 3 exactly, which stops at k = 8 where 70 episodes remain
    # at risk. Beyond that the estimate is driven by a handful of episodes and swings between
    # 0.5 and 1.0; showing it solid would misrepresent its precision.
    MIN_AT_RISK = 70
    ax = axes[1]
    ks = np.arange(1, 25)
    at_risk = np.array([(L >= k).sum() for k in ks])
    cont = np.array([(L > k).sum() / n if n else np.nan for k, n in zip(ks, at_risk)])
    ok = at_risk >= MIN_AT_RISK
    ax.plot(ks[ok], cont[ok], "o-", color=DARK, ms=3, lw=1.2,
            label="continuation probability")
    ax.plot(ks[~ok], cont[~ok], "o", color="0.8", ms=2, label=f"fewer than {MIN_AT_RISK} at risk")
    kmax = int(ks[ok].max())
    ax.axvline(kmax + .5, color="0.8", lw=.8)
    # a straight line through the well-supported range, to show the direction without implying
    # that a linear model in k is the right description
    b, a = np.polyfit(ks[ok], cont[ok], 1)
    ax.plot(ks[ok], a + b * ks[ok], color=MID, lw=.9, ls="--", label="linear trend")
    ax.annotate(f"{cont[0]:.2f}", (1, cont[0] + .05), fontsize=6, color=DARK, ha="center")
    ax.annotate(f"{cont[kmax - 1]:.2f}", (kmax, cont[kmax - 1] - .07), fontsize=6, color=DARK,
                ha="center")
    ax.set_ylim(0, 1.05); ax.set_xlabel("episode age k (h)")
    ax.set_ylabel("P(L > k | L $\\geq$ k)")
    ax.legend(fontsize=5.5, frameon=False, loc="lower left")
    ax.set_title("(b)", fontsize=8, loc="left")
    print(f"  continuation probability shown to k={kmax} (at risk {at_risk[ok][-1]}); "
          f"{cont[0]:.3f} -> {cont[kmax-1]:.3f}, slope {b:+.4f}/h")

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
    ax.set_title("(c)", fontsize=8, loc="left")
    ax.set_xlim(0, max(means) * 1.3)

    fig.tight_layout()
    fig.savefig(OUT / "fig2_episodes.png")
    plt.close(fig)
    print("fig2 written; mean vs median survival:",
          {n: (round(m, 2), round(md, 2)) for n, m, md in zip(names, means, meds)})


# --------------------------------------------------------------------------- Fig 3
def _roc(task: str) -> tuple:
    """Pooled (H, F, base rate) over all decision thresholds, from saved test-year predictions."""
    # rolling730 is the scheme of Section 5.4; fall back only if it has not been generated
    for sub in ("rolling730", "monthly", ""):
        preds = sorted((PROJECT_ROOT / "outputs" / "preds" / sub).glob(f"{task}_*.parquet"))
        if preds:
            break
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
    ax.annotate("generator: counterexample\n(Section 7.6)", (4.2, -.30), fontsize=5.5,
                color="0.35", ha="center")
    ax.axhline(0, color="0.6", lw=.8)

    # The flexible load's alpha comes from load_shifting.csv, NOT from costloss_pooled.csv.
    # The pooled file still carries the earlier flexible-load formulation that run_shifting.py
    # replaced -- no energy balance, no displaced counterfactual -- and its alphas (0.46 at a
    # 5 EUR/MWh shift cost, 1.39 at 15) are an order of magnitude above the measured ones. An
    # earlier version of this figure plotted them, which put a superseded model on the same axis
    # as the paper's results.
    ls = pd.read_csv(PROJECT_ROOT / "outputs" / "tables" / "load_shifting.csv")
    ls = ls[(ls.days_acted > 0) & np.isfinite(ls.alpha) & (ls.alpha > 0)]
    if len(ls):
        lo, hi = ls.alpha.min(), ls.alpha.max()
        ax.axvspan(lo, hi, color=DARK, alpha=.13, lw=0)
        # The band sits at alpha < 0.15, hard against the axis, so the label goes in the empty
        # lower-left rather than beside it: an inline label collides with the Christensen
        # marker at 1/3, and a leader line to it crosses both value curves.
        ax.annotate(f"shaded: flexible load, Section 7.5\nmeasured $\\alpha$ = "
                    f"{lo:.2f}–{hi:.2f}", (0.30, -0.30), ha="left", va="center",
                    fontsize=5.5, color="0.2")

    YLO = -0.5
    for _, r in pooled.iterrows():
        a = r["alpha"]
        if a <= 0 or r["actor"] != "premium_generator":
            continue                              # the load is shown as the band above
        row = be[(be.actor == r["actor"]) & (be.param == r["param"])]
        H, F, s = row["H"].mean(), row["F"].mean(), row["base_rate"].mean()
        eref = min(s, (1 - s) * a)
        V = 1 - ((1 - H) * s + F * (1 - s) * a) / eref if eref > 0 else np.nan
        col = MID
        lab = "gen, ref %g" % r["param"]
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
