# Negative electricity prices and the market premium in Germany

Replication package for **"Short episodes, asymmetric payoffs: negative electricity prices and
the market premium in Germany"** by Suho Ahn (School of Business and Technology Management,
KAIST).

The paper measures how long negative-price episodes last in the German day-ahead market,
forecasts the event that suspends the renewable market premium under § 51 of the Renewable
Energy Sources Act, and asks what such a forecast is worth to the parties who might act on it.

## What is here

```
src/                       feature and target construction, data retrieval, the persistence model
experiments/01 .. 14/      one directory per numbered experiment; each script's docstring states
                           its purpose and, where applicable, its pre-registered decision rule
data/raw/                  the exact data vintage used in the paper (parquet)
data/metadata/             retrieval timestamps and series lengths
outputs/tables/            every table in the paper, as generated
outputs/figures/           Figures 1-4
```

## Data

All series are public. Day-ahead prices, and day-ahead forecasts of load and of solar, onshore
wind and offshore wind generation for the German-Luxembourg (DE-LU) bidding zone, were retrieved
on **11 August 2026** from the Energy-Charts API operated by Fraunhofer ISE
(`api.energy-charts.info`), which republishes ENTSO-E Transparency Platform data under a
**CC BY 4.0** licence. Realised generation is included for one purpose only: it is the
deliberate oracle in the leakage audit of Section 6.3.

`data/metadata/energy_charts_retrieval.json` records the retrieval timestamp and the length of
each series.

**The archived parquet files are the experiment.** Re-running the download script will fetch the
current vintage of the same series, which will differ from what the paper used. Use the archived
files to reproduce the paper; use the script only if you want a later vintage.

## Environment

Python 3.13. Install dependencies with:

```
pip install -r requirements.txt
```

One dependency needs a note. The LEAR benchmark uses `epftoolbox`, the reference implementation
by Lago, Marcjasz, De Schutter and Weron, installed from GitHub:

```
pip install --no-deps --ignore-requires-python git+https://github.com/jeslago/epftoolbox.git
```

`--no-deps` avoids pulling TensorFlow, which the LEAR model does not use. Because that package's
`models/__init__.py` imports the TensorFlow-based DNN unconditionally,
`experiments/10_lear/run_lear_benchmark.py` loads the LEAR module directly by file path.
**No package source is modified.**

## Reproducing the paper

The scripts are not independent; later ones read earlier outputs. Run in this order:

| # | Command | Produces | Approx. runtime |
|---|---|---|---|
| 1 | `python src/data/download_energy_charts.py` | `data/raw/` (skip to use the archived vintage) | 20 min |
| 2 | `python experiments/02_episodes/analyze_episodes.py` | Tables 3-4, Section 4 | 1 min |
| 3 | `python experiments/07_leakage_audit/run_leakage_audit.py` | leakage audit, Section 6.3 | 5 min |
| 4 | `python experiments/08_infoset_dm/run_infoset_dm.py --recal month` | Table 6, DM tests | 35 min |
| 5 | `python experiments/10_lear/run_lear_benchmark.py --forecast-only` | LEAR point forecasts | 2 h |
| 6 | `python experiments/10_lear/run_lear_benchmark.py --no-exog --out-suffix _noexog --forecast-only` | LEAR without exogenous inputs | 2 h |
| 7 | `python experiments/10_lear/score_lear.py` and `--suffix _noexog` | Tables 7-8 | 2 min |
| 8 | `python experiments/11_recalibration/run_recal.py` | recalibration diagnosis | 20 min |
| 9 | `python experiments/09_costloss/run_costloss.py` | Table 11 | 15 min |
| 10 | `python experiments/13_load_only/run_load_only.py` | Table 12 | 20 min |
| 11 | `python experiments/14_load_shifting/run_shifting.py` | Tables 9-10 | 2 min |
| 12 | `python experiments/12_figures/make_figures.py` | Figures 1-4 | 1 min |

About five hours end to end on six CPU cores, of which four are the two LEAR passes. **No GPU is
required.** Steps 5 and 6 checkpoint to `outputs/preds/lear_ckpt*.parquet` and resume
automatically if interrupted.

Steps 1 and 4 write intermediate per-observation predictions to `outputs/preds/`, which is not
tracked here because it is large and fully regenerable.

## Determinism

All estimators are seeded (`SEED = 42`). The neural experiment in `experiments/05_diagnosis/`
runs three seeds and reports the spread rather than a single value.

## Notes on reading the code

Several scripts record a decision rule in their module docstring that was fixed **before** the
run, and report the outcome against it whether or not it favoured the hypothesis. See
`experiments/05_diagnosis/run_lookback_diagnosis.py`, `experiments/08_infoset_dm/`,
`experiments/11_recalibration/` and `experiments/13_load_only/`.

Two errors found during the work are documented in the code as well as in the paper:
`experiments/09_costloss/run_costloss.py` explains a train/validation leak in the threshold
tuner, and `src/features.py` explains why the autoregressive benchmark was initially
handicapped. Both are described in Section 8.1 of the paper.

## Licence

Code: MIT (see `LICENSE`).
Data: CC BY 4.0, Fraunhofer ISE / Energy-Charts, republishing ENTSO-E Transparency data. See
`data/README.md`.

## Citation

See `CITATION.cff`, or cite the archived release DOI.
