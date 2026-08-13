"""Feature and target construction for negative-price episode forecasting.

Information set (strict): everything is known at day-ahead gate closure, 12:00 CET on
D-1, for every hour of day D.
- day-ahead forecasts for solar / wind onshore / wind offshore / load are published
  before gate closure -> usable for all hours of D
- price history is usable only up to the last hour observed before gate closure on D-1;
  to stay conservative we use D-2 and earlier for same-hour lags
- ACTUAL generation/load is NEVER used (it is not known ex ante)

Targets
- y_neg   : price < 0 at hour t                       (hourly event)
- y_run4  : hour t belongs to a negative run >= 4 h    (EEG premium-suspension relevance)
- y_day4  : day D contains a negative run >= 4 h       (daily, the money question)
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "data" / "raw"
EEG_RUN_H = 4


def load_hourly() -> pd.DataFrame:
    price = pd.read_parquet(RAW / "price_de_lu.parquet")["price"]
    price = price[~price.index.duplicated(keep="first")].resample("h").mean()
    out = {"price": price}
    for pt in ("solar", "wind_onshore", "wind_offshore", "load"):
        s = pd.read_parquet(RAW / f"fc_{pt}.parquet")[f"{pt}_fc"]
        out[f"{pt}_fc"] = s[~s.index.duplicated(keep="first")].resample("h").mean()
    df = pd.DataFrame(out).sort_index()
    return df[df["price"].notna()]


def run_lengths(flag: pd.Series) -> pd.Series:
    """For each timestamp, the length of the contiguous True-run it belongs to (0 if False)."""
    f = flag.fillna(False).astype(bool)
    grp = (f != f.shift()).cumsum()
    lengths = f.groupby(grp).transform("sum")
    return lengths.where(f, 0).astype(int)


def build(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = df.copy()
    idx = d.index
    d["day"] = idx.normalize()

    # --- day-ahead forecast features (known ex ante) ---
    d["res_fc"] = d[["solar_fc", "wind_onshore_fc", "wind_offshore_fc"]].sum(axis=1)
    d["residual_load"] = d["load_fc"] - d["res_fc"]
    d["res_share"] = d["res_fc"] / d["load_fc"].replace(0, np.nan)
    d["solar_share"] = d["solar_fc"] / d["load_fc"].replace(0, np.nan)
    d["wind_share"] = (d["wind_onshore_fc"] + d["wind_offshore_fc"]) / d["load_fc"].replace(0, np.nan)
    for w in (1, 3, 6):
        d[f"resload_ramp_{w}h"] = d["residual_load"].diff(w)
        d[f"res_ramp_{w}h"] = d["res_fc"].diff(w)
    # rolling window of oversupply pressure within the day-ahead profile
    for w in (3, 6, 12):
        d[f"resload_min_{w}h"] = d["residual_load"].rolling(w, min_periods=1).min()
        d[f"res_share_max_{w}h"] = d["res_share"].rolling(w, min_periods=1).max()

    # load-only analogues of the pressure features above. The day-ahead LOAD forecast is
    # guaranteed by Art. 6(1)(b) of Regulation (EU) 543/2013 to precede gate closure by at
    # least two hours, whereas the wind/solar deadline in Art. 14(2)(d) falls after it. These
    # let us measure how much of the result survives on legally guaranteed data alone.
    for w in (1, 3, 6):
        d[f"load_ramp_{w}h"] = d["load_fc"].diff(w)
    for w in (3, 6, 12):
        d[f"load_min_{w}h"] = d["load_fc"].rolling(w, min_periods=1).min()

    # daily shape statistics of the forecast (all known before gate closure)
    g = d.groupby("day")
    for col, aggs in (("residual_load", ["min", "mean", "max"]),
                      ("res_share", ["max", "mean"]),
                      ("solar_fc", ["max"]), ("res_fc", ["max", "mean"]),
                      ("load_fc", ["min", "mean", "max"])):
        agg = g[col].agg(aggs)
        agg.columns = [f"{col}_day{a}" for a in aggs]
        d = d.join(agg, on="day")
    d["resload_day_range"] = d["residual_load_daymax"] - d["residual_load_daymin"]
    # hours of the day whose forecast residual load is in the lowest decile of the year-to-date
    d["hours_low_resload_day"] = g["residual_load"].transform(
        lambda s: (s < s.quantile(0.25)).sum())

    # --- autoregressive features, D-1 information set (the CORRECT one) ---
    # The day-ahead auction for delivery day D-1 clears at 12:00 CET on D-2 and results are
    # published ~12:42 on D-2. Therefore at gate closure for day D (12:00 CET on D-1) all 24
    # hourly prices of D-1 are public. Using only D-2 lags (the block below) is strictly
    # conservative and HANDICAPS the autoregressive benchmark. Both sets are built so the
    # information-set choice can be measured rather than assumed; see experiments/08_infoset_dm.
    d["price_h_d1"] = d["price"].shift(24)
    d["price_h_d3"] = d["price"].shift(72)
    d["neg_h_d1"] = (d["price_h_d1"] < 0).astype(float)
    d["neg_rate_7d_d1"] = (d["price"] < 0).astype(float).shift(24).rolling(168).mean()
    d["neg_rate_30d_d1"] = (d["price"] < 0).astype(float).shift(24).rolling(720).mean()

    # whole-day summaries of D-1, joined onto every hour of D (all published before closure)
    prev = d.groupby("day").agg(
        d1_price_mean=("price", "mean"),
        d1_price_min=("price", "min"),
        d1_price_max=("price", "max"),
        d1_n_neg=("price", lambda s: float((s < 0).sum())),
        # within-day max negative run on D-1; runs crossing midnight are truncated, which is
        # acceptable for a feature (it is not the target definition)
        d1_max_run=("price", lambda s: float(run_lengths(s < 0).max())),
    )
    prev.index = prev.index + pd.Timedelta(days=1)  # shift D-1's summary onto day D
    d = d.join(prev, on="day")

    # --- autoregressive features (conservative: D-2 and earlier) ---
    d["price_h_d2"] = d["price"].shift(48)
    d["price_h_d7"] = d["price"].shift(168)
    d["neg_h_d2"] = (d["price_h_d2"] < 0).astype(float)
    d["neg_rate_7d"] = (d["price"] < 0).astype(float).shift(48).rolling(168).mean()
    d["neg_rate_30d"] = (d["price"] < 0).astype(float).shift(48).rolling(720).mean()
    d["price_mean_d2"] = d["price"].shift(48).rolling(24).mean()
    d["price_min_d2"] = d["price"].shift(48).rolling(24).min()

    # --- calendar ---
    h, dow, mon = idx.hour, idx.dayofweek, idx.month
    d["sin_h"], d["cos_h"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    d["sin_dow"], d["cos_dow"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    d["sin_doy"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    d["cos_doy"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)
    d["is_weekend"] = (dow >= 5).astype(float)
    d["hour"] = h

    # --- targets ---
    neg = d["price"] < 0
    d["y_neg"] = neg.astype(int)
    rl = run_lengths(neg)
    d["run_len"] = rl
    d["y_run4"] = ((rl >= EEG_RUN_H) & neg).astype(int)
    # § 51 EEG is layered by plant vintage and several regimes are in force at once, so no
    # single run length is "the" rule. Over our sample the 2016-20 cohort faced six hours
    # throughout; the 2021-22 cohort four; the 2023 cohort four in 2023 and three from 2024;
    # and plants from 25 Feb 2025 face no run-length condition at all, which is y_neg.
    # We therefore carry every threshold the statute has used.
    for h in (2, 3, 6):
        d[f"y_run{h}"] = ((rl >= h) & neg).astype(int)
    day_flag = d.groupby("day")["y_run4"].transform("max")
    d["y_day4"] = day_flag.astype(int)

    daily = d.groupby("day").agg(y_day4=("y_day4", "max"),
                                 n_neg=("y_neg", "sum"),
                                 max_run=("run_len", "max")).reset_index()
    return d, daily


FEATURES_FC = [
    "solar_fc", "wind_onshore_fc", "wind_offshore_fc", "load_fc", "res_fc",
    "residual_load", "res_share", "solar_share", "wind_share",
    "resload_ramp_1h", "resload_ramp_3h", "resload_ramp_6h",
    "res_ramp_1h", "res_ramp_3h", "res_ramp_6h",
    "resload_min_3h", "resload_min_6h", "resload_min_12h",
    "res_share_max_3h", "res_share_max_6h", "res_share_max_12h",
    "residual_load_daymin", "residual_load_daymean", "residual_load_daymax",
    "res_share_daymax", "res_share_daymean", "solar_fc_daymax",
    "res_fc_daymax", "res_fc_daymean", "resload_day_range", "hours_low_resload_day",
]
FEATURES_AR = ["price_h_d2", "price_h_d7", "neg_h_d2", "neg_rate_7d", "neg_rate_30d",
               "price_mean_d2", "price_min_d2"]
# The correct day-ahead information set: everything above plus the D-1 price vector, which is
# public at gate closure. FEATURES_AR alone understates what an autoregressive benchmark can do.
FEATURES_AR_D1 = FEATURES_AR + [
    "price_h_d1", "price_h_d3", "neg_h_d1", "neg_rate_7d_d1", "neg_rate_30d_d1",
    "d1_price_mean", "d1_price_min", "d1_price_max", "d1_n_neg", "d1_max_run",
]
FEATURES_CAL = ["sin_h", "cos_h", "sin_dow", "cos_dow", "sin_doy", "cos_doy", "is_weekend"]

# Article 6(1)(b) of Regulation (EU) 543/2013 requires the day-ahead LOAD forecast to be
# published at least two hours before gate closure, so its presence in the day-ahead
# information set is guaranteed by law. Article 14(2)(d) sets the deadline for the wind and
# solar forecast at 17:00 on D-1, which is AFTER the 12:00 gate; in practice it is published
# earlier, but the Regulation does not guarantee it. This restricted set contains only the
# load-derived features and is used to bound how much of the result depends on the renewable
# forecast whose pre-gate availability we cannot yet document from a primary source.
FEATURES_FC_LOAD_ONLY = [
    "load_fc",
    "load_ramp_1h", "load_ramp_3h", "load_ramp_6h",
    "load_min_3h", "load_min_6h", "load_min_12h",
    "load_fc_daymin", "load_fc_daymean", "load_fc_daymax",
]
