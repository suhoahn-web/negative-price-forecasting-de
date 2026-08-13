"""Independent reproduction of the negative-price pilot.

Checks the claim that day-ahead RES/load forecasts beat the strongest autoregressive
baseline out of sample for predicting negative-price hours.

Protocol (strict, no look-ahead):
- Features known at day-ahead gate closure (12:00 CET on D-1) for every hour of day D:
    * day-ahead forecasts: solar, wind onshore, wind offshore, load, and derived
      residual load = load - (solar + wind_on + wind_off), renewable share, ramps
    * calendar: hour, weekday, month, holiday-ish proxies (cyclical encodings)
    * AR price history: prices up to and including the last hour observable before gate
      closure on D-1 (NOT same-day prices)
- Train 2023, test 2024 (matching the reported pilot), plus a rolling-origin variant.
- Metrics: PR-AUC (primary; no-skill floor = base rate), ROC-AUC, Brier.

Outputs: outputs/pilot_results.csv and a printed table.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW = PROJECT_ROOT / "data" / "raw"
OUT = PROJECT_ROOT / "outputs"

GATE_HOUR_UTC = 11  # 12:00 CET ~ 11:00 UTC (winter); conservative for CEST too


def load_data(freq: str = "h") -> pd.DataFrame:
    """Hourly panel. Forecasts are published at 15-min resolution throughout;
    prices are hourly until the SDAC 15-min market-time-unit change on 2025-10-01
    and quarter-hourly afterwards. Both are resampled to hourly means so the panel
    has one consistent resolution; the break date is recorded in the design doc.
    """
    price = pd.read_parquet(RAW / "price_de_lu.parquet")["price"]
    price = price[~price.index.duplicated(keep="first")].resample(freq).mean()
    frames = {"price": price}
    for pt in ("solar", "wind_onshore", "wind_offshore", "load"):
        f = RAW / f"fc_{pt}.parquet"
        if f.exists():
            s = pd.read_parquet(f)[f"{pt}_fc"]
            s = s[~s.index.duplicated(keep="first")].resample(freq).mean()
            frames[f"{pt}_fc"] = s
    df = pd.DataFrame(frames).sort_index()
    return df[df["price"].notna()]


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    d = df.copy()
    d["date"] = d.index.date

    fc_cols = [c for c in d.columns if c.endswith("_fc")]
    res_parts = [c for c in ("solar_fc", "wind_onshore_fc", "wind_offshore_fc") if c in d]
    if res_parts:
        d["res_fc"] = d[res_parts].sum(axis=1)
        if "load_fc" in d:
            d["residual_load"] = d["load_fc"] - d["res_fc"]
            d["res_share"] = d["res_fc"] / d["load_fc"].replace(0, np.nan)
        d["res_ramp_1h"] = d["res_fc"].diff()
        d["res_ramp_3h"] = d["res_fc"].diff(3)

    # daily aggregates of the forecast profile (known ex ante for the whole day)
    if "res_share" in d:
        daily = d.groupby("date")["res_share"].agg(["max", "mean"])
        daily.columns = ["res_share_daymax", "res_share_daymean"]
        d = d.join(daily, on="date")
    if "residual_load" in d:
        daily2 = d.groupby("date")["residual_load"].agg(["min", "mean"])
        daily2.columns = ["resload_daymin", "resload_daymean"]
        d = d.join(daily2, on="date")

    # AR price history: last value observable before gate closure on D-1.
    # Shift by 24h then take the value at/below the gate hour of the previous day.
    prev_day_gate = (d["price"]
                     .where(d.index.hour <= GATE_HOUR_UTC)
                     .groupby(d["date"]).last())
    prev_day_gate.index = pd.to_datetime(prev_day_gate.index)
    d["price_gate_d1"] = pd.Series(
        d.index.normalize().tz_localize(None) - pd.Timedelta(days=1),
        index=d.index).map(prev_day_gate)

    # same hour, D-1 and D-7 (fully observable before gate closure? D-1 same hour is NOT
    # for hours after the gate, so use D-2 to stay strictly safe)
    d["price_h_d2"] = d["price"].shift(48)
    d["price_h_d7"] = d["price"].shift(168)
    d["neg_h_d2"] = (d["price_h_d2"] < 0).astype(float)
    d["neg_rate_7d"] = (d["price"] < 0).astype(float).shift(48).rolling(168).mean()

    # calendar (cyclical)
    h = d.index.hour
    dow = d.index.dayofweek
    mon = d.index.month
    d["sin_h"], d["cos_h"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    d["sin_dow"], d["cos_dow"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    d["sin_m"], d["cos_m"] = np.sin(2 * np.pi * mon / 12), np.cos(2 * np.pi * mon / 12)
    d["is_weekend"] = (dow >= 5).astype(float)

    y = (d["price"] < 0).astype(int)
    return d, y


CAL = ["sin_h", "cos_h", "sin_dow", "cos_dow", "sin_m", "cos_m", "is_weekend"]
AR = ["price_gate_d1", "price_h_d2", "price_h_d7", "neg_h_d2", "neg_rate_7d"]
FC = ["solar_fc", "wind_onshore_fc", "wind_offshore_fc", "load_fc", "res_fc",
      "residual_load", "res_share", "res_ramp_1h", "res_ramp_3h",
      "res_share_daymax", "res_share_daymean", "resload_daymin", "resload_daymean"]


def score(y_true, p) -> dict:
    return {"PR_AUC": float(average_precision_score(y_true, p)),
            "ROC_AUC": float(roc_auc_score(y_true, p)),
            "Brier": float(brier_score_loss(y_true, p))}


def fit_logit(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=3000, C=1.0)
    m.fit(sc.transform(Xtr), ytr)
    return m.predict_proba(sc.transform(Xte))[:, 1]


def run_split(d, y, train_years, test_year) -> list[dict]:
    tr = d.index.year.isin(train_years)
    te = d.index.year == test_year
    rows = []
    base = float(y[te].mean())

    # B0 climatology: month x hour positive rate from training years
    clim = y[tr].groupby([d.index[tr].month, d.index[tr].hour]).mean()
    p_clim = pd.MultiIndex.from_arrays([d.index[te].month, d.index[te].hour])
    p0 = clim.reindex(p_clim).fillna(base).to_numpy()
    rows.append({"model": "B0 climatology", **score(y[te], p0)})

    # B1 persistence: negative at same hour 2 days ago
    p1 = d.loc[te, "neg_h_d2"].fillna(base).to_numpy() * 0.98 + 0.01
    rows.append({"model": "B1 persistence(d-2)", **score(y[te], p1)})

    feature_sets = {
        "B2 calendar only": CAL,
        "B3 calendar + AR price": CAL + AR,
        "T1 + DA forecasts": CAL + AR + FC,
        "T2 DA forecasts only": CAL + FC,
    }
    for name, cols in feature_sets.items():
        cols = [c for c in cols if c in d.columns]
        sub = d[cols]
        ok_tr = tr & sub.notna().all(axis=1).to_numpy() & y.notna().to_numpy()
        ok_te = te & sub.notna().all(axis=1).to_numpy()
        if ok_tr.sum() < 500 or ok_te.sum() < 200:
            continue
        p = fit_logit(sub[ok_tr].to_numpy(), y[ok_tr].to_numpy(), sub[ok_te].to_numpy())
        rows.append({"model": name, "n_train": int(ok_tr.sum()), "n_test": int(ok_te.sum()),
                     **score(y[ok_te], p)})

    for r in rows:
        r.update(test_year=test_year, base_rate=base,
                 lift=r["PR_AUC"] / base if base > 0 else np.nan)
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_data()
    print("loaded:", df.shape, df.index.min(), "->", df.index.max())
    print("columns:", list(df.columns))
    print("negative-price share by year:")
    print((df["price"] < 0).groupby(df.index.year).mean().round(4).to_string())

    d, y = build_features(df)
    rows = []
    rows += run_split(d, y, [2023], 2024)                      # the reported pilot
    for ty in range(2021, 2026):                                # rolling origin
        yrs = list(range(2019, ty))
        if len(yrs) >= 2:
            rows += [{**r, "setup": f"expanding<{ty}"} for r in run_split(d, y, yrs, ty)]

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "pilot_results.csv", index=False)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("\n=== PILOT (train 2023 -> test 2024) ===")
    p = res[(res.test_year == 2024) & (res.setup.isna() if "setup" in res else True)]
    print(p[["model", "base_rate", "PR_AUC", "ROC_AUC", "Brier", "lift"]].round(4).to_string(index=False))
    if "setup" in res:
        print("\n=== ROLLING ORIGIN (expanding window) ===")
        r = res[res.setup.notna()]
        piv = r.pivot_table(index="model", columns="test_year", values="PR_AUC")
        print(piv.round(3).to_string())


if __name__ == "__main__":
    sys.exit(main())
