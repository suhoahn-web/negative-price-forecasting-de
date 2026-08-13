# Data

## Source and licence

All series in `raw/` were retrieved from the **Energy-Charts API operated by Fraunhofer ISE**
(`api.energy-charts.info`), which republishes data from the **ENTSO-E Transparency Platform**.

**Licence: CC BY 4.0** (Energy-Charts / Fraunhofer ISE). Redistribution is permitted with
attribution, which this file provides.

Bidding zone: **DE-LU** (Germany–Luxembourg).
Retrieved: **11 August 2026**. Exact timestamps and per-series lengths are in
`metadata/energy_charts_retrieval.json`.

## Files

| File | Contents | Role in the paper |
|---|---|---|
| `raw/price_de_lu.parquet` | day-ahead clearing price | the target is derived from this |
| `raw/fc_solar.parquet` | day-ahead solar generation forecast | predictor |
| `raw/fc_wind_onshore.parquet` | day-ahead onshore wind forecast | predictor |
| `raw/fc_wind_offshore.parquet` | day-ahead offshore wind forecast | predictor |
| `raw/fc_load.parquet` | day-ahead load forecast | predictor |
| `raw/actuals_de.parquet` | realised generation | **not a predictor.** Used once, as the deliberate oracle in the leakage audit of Section 6.3, test T3 |

The four forecast series are retrieved with the API's `day-ahead` forecast type, which is a
distinct vintage from the intraday and current revisions of the same quantities. Section 6.3 of
the paper reports the tests establishing that it behaves as an ex-ante forecast rather than a
relabelled outturn.

## Frequency

The price series is mixed-frequency: the German market moved from hourly to quarter-hourly
day-ahead products during the sample. All series are converted to UTC and aggregated to hourly
means. This matches the statute: under § 100(45) EEG an hour counts as negative for the
run-length regimes if the mean of its four quarter-hourly prices is negative.

## Reproducibility caveat

`src/data/download_energy_charts.py` will fetch the **current** vintage of these series, which
will not equal the archived files. The archived parquet files are the ones the paper uses.
