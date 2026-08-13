"""Download German (DE-LU) day-ahead prices and day-ahead RES/load forecasts.

Source: Energy-Charts API (Fraunhofer ISE), https://api.energy-charts.info
No API key or registration required. Data licensed CC BY 4.0; underlying data from
Bundesnetzagentur | SMARD.de and ENTSO-E.

IMPORTANT (leakage): only the "day-ahead" forecast_type is used for predictors.
These are published before day-ahead market gate closure (12:00 CET D-1), so they
are genuinely available ex ante for every hour of day D. Actuals are downloaded
separately and used ONLY to construct targets and for descriptive analysis.

Raw responses are cached under negprice/data/raw/ and treated as read-only.
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW = PROJECT_ROOT / "data" / "raw"
META = PROJECT_ROOT / "data" / "metadata"
BASE = "https://api.energy-charts.info"

BZN = "DE-LU"
COUNTRY = "de"
YEARS = list(range(2019, 2026))
FORECASTS = [("solar", "day-ahead"), ("wind_onshore", "day-ahead"),
             ("wind_offshore", "day-ahead"), ("load", "day-ahead")]
PAUSE = 3.0  # the API rate-limits aggressively (HTTP 429)


def _get(path: str, params: dict, retries: int = 5) -> dict:
    for attempt in range(retries):
        r = requests.get(f"{BASE}{path}", params=params, timeout=180)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            wait = PAUSE * (attempt + 2)
            print(f"    429 rate limited, waiting {wait:.0f}s", flush=True)
            time.sleep(wait)
            continue
        raise RuntimeError(f"{path} {params} -> HTTP {r.status_code}: {r.text[:200]}")
    raise RuntimeError(f"{path} {params} -> repeated 429")


def to_series(j: dict, key: str, name: str) -> pd.Series:
    idx = pd.to_datetime(j["unix_seconds"], unit="s", utc=True)
    return pd.Series(j[key], index=idx, name=name, dtype="float64")


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    META.mkdir(parents=True, exist_ok=True)
    meta = {"source": BASE, "bzn": BZN, "license": "CC BY 4.0 (Energy-Charts / Fraunhofer ISE)",
            "retrieval_utc": datetime.now(timezone.utc).isoformat(), "series": {}}

    # --- day-ahead prices ---
    prices = []
    for y in YEARS:
        print(f"price {y}", flush=True)
        j = _get("/price", {"bzn": BZN, "start": f"{y}-01-01", "end": f"{y}-12-31"})
        prices.append(to_series(j, "price", "price"))
        time.sleep(PAUSE)
    price = pd.concat(prices).sort_index()
    price = price[~price.index.duplicated(keep="first")]
    price.to_frame().to_parquet(RAW / "price_de_lu.parquet")
    meta["series"]["price"] = {"n": int(price.notna().sum()),
                               "start": str(price.index.min()), "end": str(price.index.max())}

    # --- day-ahead forecasts (predictors) ---
    for pt, ft in FORECASTS:
        parts = []
        for y in YEARS:
            print(f"{pt} {ft} {y}", flush=True)
            try:
                j = _get("/public_power_forecast",
                         {"country": COUNTRY, "production_type": pt,
                          "forecast_type": ft, "start": f"{y}-01-01", "end": f"{y}-12-31"})
            except RuntimeError as exc:
                print(f"    skipped: {exc}", flush=True)
                time.sleep(PAUSE)
                continue
            if not j.get("unix_seconds"):
                print("    empty", flush=True)
                time.sleep(PAUSE)
                continue
            parts.append(to_series(j, "forecast_values", f"{pt}_fc"))
            time.sleep(PAUSE)
        if not parts:
            print(f"  !! no data for {pt}", flush=True)
            continue
        s = pd.concat(parts).sort_index()
        s = s[~s.index.duplicated(keep="first")]
        s.to_frame().to_parquet(RAW / f"fc_{pt}.parquet")
        meta["series"][f"{pt}_fc"] = {"n": int(s.notna().sum()),
                                      "start": str(s.index.min()), "end": str(s.index.max())}
        print(f"  -> {pt}: {s.notna().sum()} obs {s.index.min()} .. {s.index.max()}", flush=True)

    (META / "energy_charts_retrieval.json").write_text(json.dumps(meta, indent=2))
    print("\nmetadata ->", META / "energy_charts_retrieval.json")


if __name__ == "__main__":
    sys.exit(main())
