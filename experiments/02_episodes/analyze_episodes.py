"""Episode / run-length analysis of German negative-price events.

This is the paper's core descriptive contribution: nobody has characterised the
DURATION distribution of negative-price episodes, and the EEG market-premium rule
makes the 4-hour run length an economically binding threshold.

Produces:
- run-length distribution and survival curve per year
- geometric-hazard half-life of the negative state (target our learned rho must recover)
- comparison against the half-life of the price LEVEL (AR(1)) and against
  threshold-based spike definitions (fixed vs rolling) to show threshold sensitivity
- share of episodes reaching the EEG 4-hour premium-suspension threshold
Outputs: outputs/tables/episode_stats.csv, outputs/figures/episode_*.png,
         experiments/02_episodes/EPISODE_NOTES.md
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "01_pilot"))
from verify_pilot import load_data  # noqa: E402

OUT_T = PROJECT_ROOT / "outputs" / "tables"
OUT_F = PROJECT_ROOT / "outputs" / "figures"
HERE = PROJECT_ROOT / "experiments" / "02_episodes"
EEG_THRESHOLD_H = 4  # 2021 EEG rule: premium suspended after 4 consecutive negative hours


def episodes(flag: pd.Series) -> pd.DataFrame:
    """Contiguous runs of True, with start/end/length (hours)."""
    f = flag.fillna(False).astype(bool).to_numpy()
    idx = flag.index
    runs = []
    i = 0
    while i < len(f):
        if f[i]:
            j = i
            while j + 1 < len(f) and f[j + 1]:
                j += 1
            runs.append({"start": idx[i], "end": idx[j], "length_h": j - i + 1})
            i = j + 1
        else:
            i += 1
    return pd.DataFrame(runs)


def geometric_half_life(lengths: np.ndarray) -> tuple[float, float]:
    """Continuation probability from a geometric fit; implied half-life in hours.

    For a geometric run-length with continuation probability p, E[L] = 1/(1-p),
    so p = 1 - 1/mean. Half-life = ln(0.5)/ln(p).
    """
    m = float(np.mean(lengths))
    if m <= 1:
        return 0.0, 0.0
    p = 1.0 - 1.0 / m
    return p, float(np.log(0.5) / np.log(p))


def empirical_hazard(lengths: np.ndarray, max_k: int = 8) -> dict:
    """P(run continues past k | reached k) for k = 1..max_k."""
    out = {}
    for k in range(1, max_k + 1):
        at_risk = (lengths >= k).sum()
        cont = (lengths > k).sum()
        out[k] = float(cont / at_risk) if at_risk >= 10 else np.nan
    return out


def ar1_half_life(price: pd.Series) -> float:
    """Half-life of the deseasonalised price level (hour-of-day/month means removed)."""
    p = price.dropna()
    d = p.groupby([p.index.month, p.index.hour]).transform("mean")
    r = (p - d).to_numpy()
    x, y = r[:-1], r[1:]
    phi = float(np.dot(x, y) / np.dot(x, x))
    return float(np.log(0.5) / np.log(phi)) if 0 < phi < 1 else np.nan


def main() -> None:
    for d in (OUT_T, OUT_F, HERE):
        d.mkdir(parents=True, exist_ok=True)
    df = load_data()
    price = df["price"]
    neg = price < 0

    lines = ["# Negative-Price Episode Analysis (auto-generated)\n",
             f"German DE-LU hourly, {price.index.min():%Y-%m-%d} to {price.index.max():%Y-%m-%d}, "
             f"n = {len(price):,}\n"]

    # --- overall episode statistics ---
    ep = episodes(neg)
    L = ep.length_h.to_numpy()
    p_cont, hl = geometric_half_life(L)
    lines.append(f"## Overall\n")
    lines.append(f"- negative hours: {int(neg.sum()):,} ({neg.mean():.2%} of hours)")
    lines.append(f"- episodes: {len(ep):,} | median {np.median(L):.0f} h | "
                 f"mean {L.mean():.2f} h | max {L.max():.0f} h")
    lines.append(f"- geometric continuation p = {p_cont:.3f} -> **half-life {hl:.2f} h**")
    lines.append(f"- price-level AR(1) half-life (deseasonalised): {ar1_half_life(price):.1f} h")
    lines.append(f"- share of episodes reaching the EEG {EEG_THRESHOLD_H}h threshold: "
                 f"**{(L >= EEG_THRESHOLD_H).mean():.1%}**")
    lines.append(f"- share of negative HOURS inside episodes >= {EEG_THRESHOLD_H}h: "
                 f"{L[L >= EEG_THRESHOLD_H].sum() / L.sum():.1%}")

    haz = empirical_hazard(L)
    lines.append("\n### Empirical continuation hazard P(L > k | L >= k)\n")
    lines.append("| k (h) | " + " | ".join(str(k) for k in haz) + " |")
    lines.append("|---" * (len(haz) + 1) + "|")
    lines.append("| hazard | " + " | ".join(
        f"{v:.2f}" if not np.isnan(v) else "-" for v in haz.values()) + " |")
    lines.append("\nA roughly flat hazard indicates an approximately geometric (memoryless) "
                 "process, i.e. an exponential decay is the correct functional form.")

    # --- per-year ---
    rows = []
    lines.append("\n## By year\n")
    lines.append("| year | neg hours | share | episodes | median | mean | half-life | "
                 f">= {EEG_THRESHOLD_H}h |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for y, g in price.groupby(price.index.year):
        if len(g) < 1000:
            continue
        n = g < 0
        e = episodes(n)
        if len(e) < 5:
            continue
        Ly = e.length_h.to_numpy()
        _, hly = geometric_half_life(Ly)
        rows.append({"year": y, "n_hours": len(g), "neg_hours": int(n.sum()),
                     "neg_share": float(n.mean()), "episodes": len(e),
                     "median_h": float(np.median(Ly)), "mean_h": float(Ly.mean()),
                     "half_life_h": hly,
                     "share_ge_eeg": float((Ly >= EEG_THRESHOLD_H).mean())})
        lines.append(f"| {y} | {int(n.sum())} | {n.mean():.2%} | {len(e)} | "
                     f"{np.median(Ly):.0f} | {Ly.mean():.2f} | {hly:.2f} | "
                     f"{(Ly >= EEG_THRESHOLD_H).mean():.1%} |")
    pd.DataFrame(rows).to_csv(OUT_T / "episode_stats.csv", index=False)

    # --- threshold sensitivity: negative vs fixed vs rolling spike thresholds ---
    lines.append("\n## Threshold sensitivity (why negative prices are the clean target)\n")
    lines.append("| definition | base rate | episodes | median | mean | half-life |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    defs = {"negative (< 0)": neg,
            "fixed > 200 EUR/MWh": price > 200,
            "rolling 30d p95": price > price.rolling(24 * 30, min_periods=240).quantile(0.95),
            "rolling 30d p99": price > price.rolling(24 * 30, min_periods=240).quantile(0.99)}
    for name, flag in defs.items():
        e = episodes(flag)
        if len(e) < 5:
            continue
        Ld = e.length_h.to_numpy()
        _, h = geometric_half_life(Ld)
        lines.append(f"| {name} | {flag.mean():.2%} | {len(e)} | {np.median(Ld):.0f} | "
                     f"{Ld.mean():.2f} | {h:.2f} |")
    lines.append("\nA FIXED threshold inflates measured persistence because it conflates the "
                 "2022 gas-crisis level shift with genuine clustering. Zero is an absolute, "
                 "non-drifting boundary requiring no threshold choice.")

    # --- figures ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(L, bins=range(1, int(L.max()) + 2), edgecolor="white")
    axes[0].axvline(EEG_THRESHOLD_H, color="crimson", ls="--",
                    label=f"EEG {EEG_THRESHOLD_H}h threshold")
    axes[0].set(xlabel="episode length (h)", ylabel="count", title="Run-length distribution")
    axes[0].legend()

    ks = np.arange(1, 25)
    surv = [(L >= k).mean() for k in ks]
    axes[1].step(ks, surv, where="post")
    axes[1].axvline(EEG_THRESHOLD_H, color="crimson", ls="--")
    axes[1].set(xlabel="k (h)", ylabel="P(L >= k)", title="Survival curve", yscale="log")

    ys = pd.DataFrame(rows).set_index("year")
    ax2 = axes[2]
    ax2.bar(ys.index, ys.neg_share * 100, alpha=0.7)
    ax2.set(xlabel="year", ylabel="negative-price hours (%)", title="Growth of the phenomenon")
    ax3 = ax2.twinx()
    ax3.plot(ys.index, ys.half_life_h, color="crimson", marker="o")
    ax3.set_ylabel("episode half-life (h)", color="crimson")
    fig.tight_layout()
    fig.savefig(OUT_F / "episode_overview.png", dpi=150)
    plt.close(fig)

    text = "\n".join(lines)
    (HERE / "EPISODE_NOTES.md").write_text(text, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(text)


if __name__ == "__main__":
    sys.exit(main())
