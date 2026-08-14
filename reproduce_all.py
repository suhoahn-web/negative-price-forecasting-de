"""Regenerate every table, figure and headline number in the paper, in one command.

    python reproduce_all.py                 everything, about five and a half hours
    python reproduce_all.py --skip-lear     everything except the two LEAR passes, about 90 min
    python reproduce_all.py --list          print the stages and exit

The stages are ordered by dependency, not by section number: later scripts read earlier outputs.
Each stage prints its own diagnostics; this file adds only the ordering, the timing and a final
check that every artefact the manuscript cites now exists on disk.

The archived data vintage in data/raw/ is used as-is. The download step is deliberately not part
of this pipeline, because re-running it fetches the current vintage of the same series and would
silently change the inputs; see README.md.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STAGES = [
    ("episodes", "experiments/02_episodes/analyze_episodes.py", [],
     "Tables 2, 3, 5, 6, 7", 1),
    ("duration", "experiments/15_duration_test/run_duration_test.py", [], "Table 4", 1),
    ("leakage", "experiments/07_leakage_audit/run_leakage_audit.py", [],
     "Table D1, Section 6.3", 5),
    ("infoset", "experiments/08_infoset_dm/run_infoset_dm.py",
     ["--recal", "month", "--window-days", "730"], "Table 8, Appendix I", 40),
    ("infoset-expanding", "experiments/08_infoset_dm/run_infoset_dm.py",
     ["--recal", "month"], "Appendix H, the expanding-window column", 40),
    ("inference", "experiments/16_inference/run_gw_inference.py", [], "Table 9", 10),
    ("inference-expanding", "experiments/16_inference/run_gw_inference.py",
     ["--preds", "monthly"], "Appendix H", 10),
    ("lear", "experiments/10_lear/run_lear_benchmark.py", ["--forecast-only"],
     "LEAR point forecasts", 120),
    ("lear-noexog", "experiments/10_lear/run_lear_benchmark.py",
     ["--no-exog", "--out-suffix", "_noexog", "--forecast-only"],
     "LEAR without exogenous inputs", 120),
    ("lear-score", "experiments/10_lear/score_lear.py", [], "Table 10", 2),
    ("lear-score-noexog", "experiments/10_lear/score_lear.py", ["--suffix", "_noexog"],
     "Table 11", 2),
    ("recalibration", "experiments/11_recalibration/run_recal.py", [], "Section 6.2", 20),
    ("calibration", "experiments/17_calibration/run_calibration.py", [],
     "Table J1, Figure 4", 2),
    ("costloss", "experiments/09_costloss/run_costloss.py", [], "Table 15", 15),
    ("load-only", "experiments/13_load_only/run_load_only.py", [], "Table 16", 20),
    ("shifting", "experiments/14_load_shifting/run_shifting.py", [], "Tables 12, 13, 14", 2),
    ("amendment", "experiments/18_amendment/run_amendment_test.py", [], "Table K1", 1),
    ("figures", "experiments/12_figures/make_figures.py", [], "Figures 1-3", 1),
]

LEAR_STAGES = {"lear", "lear-noexog"}

ARTEFACTS = [
    "outputs/tables/episode_stats.csv", "outputs/tables/duration_dependence.csv",
    "outputs/tables/infoset_comparison_monthly_w730.csv", "outputs/tables/gw_inference.csv",
    "outputs/tables/gw_inference_monthly.csv", "outputs/tables/lear_comparison.csv",
    "outputs/tables/lear_comparison_noexog.csv", "outputs/tables/recalibration.csv",
    "outputs/tables/calibration.csv", "outputs/tables/costloss_breakeven.csv",
    "outputs/tables/load_only.csv", "outputs/tables/load_shifting.csv",
    "outputs/tables/amendment_test_regression.csv",
    "outputs/figures/fig1_timeline.png", "outputs/figures/fig2_episodes.png",
    "outputs/figures/fig3_value_curve.png", "outputs/figures/fig4_calibration.png",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-lear", action="store_true",
                    help="skip the two LEAR passes, which are four of the five and a half hours")
    ap.add_argument("--only", nargs="*", help="run only the named stages")
    ap.add_argument("--list", action="store_true", help="print the stages and exit")
    a = ap.parse_args()

    stages = [s for s in STAGES
              if not (a.skip_lear and s[0] in LEAR_STAGES)
              and (not a.only or s[0] in a.only)]

    if a.list:
        total = sum(s[4] for s in STAGES)
        for name, script, args, produces, mins in STAGES:
            print(f"  {name:20} {mins:4} min  {produces}")
        print(f"\n  {'total':20} {total:4} min")
        return 0

    print(f"reproduce_all: {len(stages)} stages, "
          f"about {sum(s[4] for s in stages)} minutes\n")
    t0 = time.time()
    failed = []
    for i, (name, script, args, produces, mins) in enumerate(stages, 1):
        print(f"[{i}/{len(stages)}] {name}  ->  {produces}")
        t = time.time()
        r = subprocess.run([sys.executable, str(ROOT / script), *args], cwd=ROOT)
        el = time.time() - t
        if r.returncode:
            failed.append(name)
            print(f"          FAILED after {el:.0f}s, exit {r.returncode}\n")
        else:
            print(f"          ok, {el:.0f}s\n")

    print("=" * 62)
    missing = [p for p in ARTEFACTS if not (ROOT / p).exists()]
    if a.skip_lear:
        missing = [p for p in missing if "lear" not in p]
    for p in ARTEFACTS:
        if (ROOT / p).exists():
            continue
        print(f"  MISSING {p}")
    print(f"stages run {len(stages)}, failed {len(failed)}"
          + (f" {failed}" if failed else ""))
    print(f"artefacts present {len(ARTEFACTS) - len(missing)} of {len(ARTEFACTS)}")
    print(f"elapsed {(time.time() - t0) / 60:.1f} minutes")
    return 1 if failed or missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
