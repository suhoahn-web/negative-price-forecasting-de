# Negative electricity prices and the market premium in Germany

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21926317.svg)](https://doi.org/10.5281/zenodo.21926317)

Replication package for **"Short episodes, asymmetric payoffs: forecasting negative electricity
prices and the market premium in Germany"** by Suho Ahn (School of Business and Technology
Management, KAIST).

The paper measures how long negative-price episodes last in the German day-ahead market,
forecasts the event that suspends the renewable market premium under § 51 of the Renewable
Energy Sources Act, and asks what such a forecast is worth to the parties who might act on it.

## What is here

```
src/                       feature and target construction, data retrieval, the persistence model
experiments/01 .. 16/      one directory per numbered experiment; each script's docstring states
                           its purpose and, where applicable, its pre-specified decision rule
data/raw/                  the exact data vintage used in the paper (parquet)
data/metadata/             retrieval timestamps and series lengths
outputs/tables/            every table in the paper, as generated
outputs/figures/           Figures 1-3, plus one unnumbered diagnostic from experiment 02
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

## The estimation scheme, and why it is what it is

Classifiers are refitted **monthly** on a **rolling 730-day window**. These are two separate
choices and only one of them is about computation.

The refit schedule is monthly rather than daily as a concession to cost; `experiments/11_recalibration/`
reports the pre-specified test that fixed it. The 730-day window is not a tuning choice: the
paper's primary test is Giacomini & White's (2006) test of conditional predictive ability, whose
asymptotics require the maximum estimation sample to stay finite as the out-of-sample count
grows. An expanding window does not satisfy that, so forecasts produced by one cannot be tested
this way. `experiments/08_infoset_dm/run_infoset_dm.py` documents this in `fit_block`.

**Passing `--window-days 730` is therefore required to reproduce the paper's headline numbers.**
Omitting it produces expanding-window forecasts, which are reported in the paper only as a
robustness check (Appendix H) and which the inference is not valid for.

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

Everything at once:

```
python reproduce_all.py                 # all stages, about seven hours
python reproduce_all.py --skip-lear     # everything except the two LEAR passes, about 90 minutes
python reproduce_all.py --list          # the stages and their runtimes
```

It runs the stages in dependency order and ends by checking that every artefact the manuscript
cites exists. The stages, if you would rather run them one at a time:

| # | Command | Produces | Approx. runtime |
|---|---|---|---|
| 1 | `python src/data/download_energy_charts.py` | `data/raw/` (skip to use the archived vintage) | 20 min |
| 2 | `python experiments/02_episodes/analyze_episodes.py` | Tables 2, 3, 5, 6, 7 | 1 min |
| 3 | `python experiments/15_duration_test/run_duration_test.py` | Table 4 | 1 min |
| 4 | `python experiments/07_leakage_audit/run_leakage_audit.py` | leakage audit, Section 6.3 | 5 min |
| 5 | `python experiments/08_infoset_dm/run_infoset_dm.py --recal month --window-days 730` | Table 8 | 40 min |
| 6 | `python experiments/16_inference/run_gw_inference.py` | Table 9 | 10 min |
| 7 | `python experiments/10_lear/run_lear_benchmark.py --forecast-only` | LEAR point forecasts | 2 h |
| 8 | `python experiments/10_lear/run_lear_benchmark.py --no-exog --out-suffix _noexog --forecast-only` | LEAR without exogenous inputs | 2 h |
| 9 | `python experiments/10_lear/score_lear.py` and `--suffix _noexog` | Tables 10, 11 | 2 min |
| 10 | `python experiments/11_recalibration/run_recal.py` | recalibration diagnosis, Section 6.2 | 20 min |
| 11 | `python experiments/09_costloss/run_costloss.py` | Table 15 | 15 min |
| 12 | `python experiments/13_load_only/run_load_only.py` | Table 16 | 20 min |
| 13 | `python experiments/14_load_shifting/run_shifting.py` | Tables 12, 13, 14 | 2 min |
| 14 | `python experiments/17_calibration/run_calibration.py` | Table J1, J2, Figure 4 | 2 min |
| 15 | `python experiments/18_amendment/run_amendment_test.py` | Table K1 | 1 min |
| 16 | `python experiments/12_figures/make_figures.py` | Figures 1-3 | 1 min |

About five and a half hours end to end on six CPU cores, of which four are the two LEAR passes.
**No GPU is required.** Steps 7 and 8 checkpoint to `outputs/preds/lear_ckpt*.parquet` and resume
automatically if interrupted.

To reproduce the expanding-window robustness column of Appendix H, run step 5 again without
`--window-days` and then `python experiments/16_inference/run_gw_inference.py --preds monthly`.

Steps 1 and 5 write intermediate per-observation predictions to `outputs/preds/`, which is not
tracked here because it is large and fully regenerable.

## A note on the metric column name

The result tables carry a column called `PR_AUC`. It holds **average precision**, computed by
`sklearn.metrics.average_precision_score`, which is a precision-weighted sum of the recall
increments rather than an interpolated integral of the precision-recall curve. No script in this
package integrates that curve. The manuscript calls the quantity average precision (AP) for that
reason; the column name is left alone so that archived result files stay readable by the scripts
that wrote them.

## Determinism

All estimators are seeded (`SEED = 42`). The neural experiment in `experiments/05_diagnosis/`
runs three seeds and reports the spread rather than a single value. The stationary block
bootstrap in `experiments/16_inference/` uses the same seed and 5,000 replications.

## Notes on reading the code

Several scripts record a decision rule in their module docstring that was fixed **before** the
run, and report the outcome against it whether or not it favoured the hypothesis. See
`experiments/05_diagnosis/run_lookback_diagnosis.py`, `experiments/08_infoset_dm/`,
`experiments/11_recalibration/` and `experiments/13_load_only/`.

Three corrections made during the work are documented in the code as well as in the paper:
`experiments/09_costloss/run_costloss.py` explains a train/validation leak in the threshold
tuner; `src/features.py` explains why the autoregressive benchmark was initially handicapped by
D-2 lags; and `experiments/08_infoset_dm/run_infoset_dm.py` explains why the estimation window
changed from expanding to rolling. All three are described in Section 8.1 and Appendix G of the
paper.

## Licence

Code: MIT (see `LICENSE`).
Data: CC BY 4.0, Fraunhofer ISE / Energy-Charts, republishing ENTSO-E Transparency data. See
`data/README.md`.

## Citation

Cite the archived deposit:

> Ahn, S. (2026). *Replication package for "Short episodes, asymmetric payoffs: forecasting
> negative electricity prices and the market premium in Germany"* [Software]. Zenodo.
> https://doi.org/10.5281/zenodo.21926317

That is the **concept DOI**, and it is the one the manuscript cites. It always resolves to the
most recent archived release, so it does not go stale when the package is revised. Every release
also carries its own version DOI for anyone who needs to pin an exact snapshot; those are listed
on the Zenodo page rather than here, because a version DOI is minted at the moment a release is
cut and so cannot be written into the release it identifies. `CITATION.cff` records the concept
DOI in machine-readable form.
