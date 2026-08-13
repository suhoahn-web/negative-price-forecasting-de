"""Train the Oversupply-Pressure Persistence Network with ablations, rolling origin.

Ablations (the paper's Table 3):
  A0  no pressure encoder, no persistence, no gate   (plain sequence net)
  A1  + pressure encoder                              (pressure but no memory)
  A2  + persistence memory, no gate
  A3  + gate, no persistence
  A4  full model                                      (pressure + memory + gate)

Central scientific test: does the learned rho in A2/A4 imply a half-life near the
independently MEASURED negative-price episode half-life (3.27 h), and far from the
price-level half-life (47.6 h)?

Split: expanding window; validation = last full year before the test year (early
stopping only); test = the target year. Metrics: PR-AUC (primary), ROC-AUC, Brier.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.features import FEATURES_AR, FEATURES_CAL, build, load_hourly  # noqa: E402
from src.model import OPPN  # noqa: E402

# Sequence features: the day-ahead forecast profile. Index 0 is the anchor used by the
# pressure encoder's skip connection: NEGATIVE residual load, so larger = more oversupply.
SEQ_FEATURES = ["neg_residual_load", "res_share", "solar_share", "wind_share",
                "res_fc", "load_fc", "resload_ramp_1h", "resload_ramp_3h"]
STATIC_FEATURES = FEATURES_CAL + FEATURES_AR + [
    "residual_load_daymin", "res_share_daymax", "resload_day_range"]

VARIANTS = {
    "A0": dict(use_pressure=False, use_persistence=False, use_gate=False),
    "A1": dict(use_pressure=True, use_persistence=False, use_gate=False),
    "A2": dict(use_pressure=True, use_persistence=True, use_gate=False),
    "A3": dict(use_pressure=True, use_persistence=False, use_gate=True),
    "A4": dict(use_pressure=True, use_persistence=True, use_gate=True),
}
LOOKBACK = 24
TEST_YEARS = [2021, 2022, 2023, 2024, 2025]


def make_windows(d: pd.DataFrame, target: str, lookback: int):
    d = d.copy()
    d["neg_residual_load"] = -d["residual_load"]
    seq = d[SEQ_FEATURES].to_numpy(dtype=np.float32)
    stat = d[STATIC_FEATURES].to_numpy(dtype=np.float32)
    y = d[target].to_numpy(dtype=np.float32)
    ok = np.isfinite(seq).all(1) & np.isfinite(stat).all(1) & np.isfinite(y)

    rows = [i for i in range(lookback - 1, len(d))
            if ok[i - lookback + 1: i + 1].all()]
    rows = np.array(rows)
    X_seq = np.stack([seq[i - lookback + 1: i + 1] for i in rows])
    return X_seq, stat[rows], y[rows], d.index[rows]


def standardize(train, *others):
    flat = train.reshape(-1, train.shape[-1])
    mu, sd = flat.mean(0), flat.std(0)
    sd[sd == 0] = 1.0
    return ((train - mu) / sd, *[(o - mu) / sd for o in others])


def run(variant: str, seed: int, test_year: int, data, epochs: int, device: str) -> dict:
    X_seq, X_st, Y, index = data
    torch.manual_seed(seed)
    np.random.seed(seed)

    years = index.year
    tr, va, te = years < test_year - 1, years == test_year - 1, years == test_year
    if tr.sum() < 2000 or va.sum() < 500 or te.sum() < 500 or Y[tr].sum() < 30:
        return {}

    seq_tr, seq_va, seq_te = standardize(X_seq[tr], X_seq[va], X_seq[te])
    st_tr, st_va, st_te = standardize(X_st[tr], X_st[va], X_st[te])

    model = OPPN(n_seq_features=X_seq.shape[-1], n_static=X_st.shape[-1],
                 **VARIANTS[variant]).to(device)
    pos_weight = torch.tensor([(len(Y[tr]) - Y[tr].sum()) / max(Y[tr].sum(), 1)],
                              device=device, dtype=torch.float32)
    lossfn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    rho_p = [p for n, p in model.named_parameters() if "memory" in n]
    other = [p for n, p in model.named_parameters() if "memory" not in n]
    groups = [{"params": other, "lr": 1e-3, "weight_decay": 1e-4}]
    if rho_p:
        groups.append({"params": rho_p, "lr": 1e-2, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups)

    dl = DataLoader(TensorDataset(torch.tensor(seq_tr), torch.tensor(st_tr),
                                  torch.tensor(Y[tr])), batch_size=256, shuffle=True)
    t_va = (torch.tensor(seq_va).to(device), torch.tensor(st_va).to(device))
    t_te = (torch.tensor(seq_te).to(device), torch.tensor(st_te).to(device))

    best, best_state, bad, n_ep = -np.inf, None, 0, 0
    for _ in range(epochs):
        model.train()
        for xb, sb, yb in dl:
            out = model(xb.to(device), sb.to(device))
            loss = lossfn(out["logit"], yb.to(device))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(*t_va)["logit"]).cpu().numpy()
        s = average_precision_score(Y[va], pv)
        n_ep += 1
        if s > best + 1e-5:
            best, bad = s, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 8:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        o = model(*t_te)
        pt = torch.sigmoid(o["logit"]).cpu().numpy()
        gate = o["gate"].cpu().numpy()
        M = o["M"].cpu().numpy()

    res = {"variant": variant, "seed": seed, "test_year": test_year,
           "val_pr_auc": float(best),
           "PR_AUC": float(average_precision_score(Y[te], pt)),
           "ROC_AUC": float(roc_auc_score(Y[te], pt)),
           "Brier": float(brier_score_loss(Y[te], pt)),
           "base_rate": float(Y[te].mean()),
           "gate_mean": float(gate.mean()), "gate_std": float(gate.std()),
           "epochs": n_ep}
    res.update(model.diagnostics())
    if VARIANTS[variant]["use_persistence"] and M.std() > 0:
        res["M_corr_target"] = float(np.corrcoef(M, Y[te])[0, 1])
    if gate.std() > 0:
        res["gate_corr_target"] = float(np.corrcoef(gate, Y[te])[0, 1])
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="y_run4", choices=["y_neg", "y_run4"])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lookback", type=int, default=LOOKBACK)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    d, _ = build(load_hourly())
    data = make_windows(d, args.task, args.lookback)
    print(f"task={args.task} windows={len(data[2])} pos_rate={data[2].mean():.4f} "
          f"device={device}", flush=True)

    rows = []
    for v in args.variants:
        for ty in TEST_YEARS:
            for s in args.seeds:
                r = run(v, s, ty, data, args.epochs, device)
                if r:
                    rows.append(r)
                    print(f"  {v} {ty} s{s}: PR-AUC {r['PR_AUC']:.3f} "
                          f"HL={r.get('half_life_h')}", flush=True)

    res = pd.DataFrame(rows)
    out = PROJECT_ROOT / "outputs" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / f"oppn_{args.task}.csv", index=False)

    agg = res.groupby(["variant", "test_year"]).agg(
        PR_AUC=("PR_AUC", "mean"), HL=("half_life_h", "mean"),
        gate=("gate_mean", "mean")).reset_index()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("\n=== PR-AUC by variant and test year (seed mean) ===")
    print(agg.pivot(index="variant", columns="test_year", values="PR_AUC").round(3).to_string())
    if res["half_life_h"].notna().any():
        print("\n=== Learned half-life (hours); MEASURED episode half-life = 3.27 h ===")
        print(agg.pivot(index="variant", columns="test_year", values="HL").round(2).to_string())
        print("\nper-seed spread:")
        print(res[res.half_life_h.notna()].groupby("variant")["half_life_h"]
              .agg(["mean", "std", "min", "max"]).round(2).to_string())


if __name__ == "__main__":
    sys.exit(main())
