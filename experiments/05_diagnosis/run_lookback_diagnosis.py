"""Diagnosis: is the persistence memory redundant with the input window?

Hypothesis for the flat ablation in experiment 04: with a 24 h input window the network
already sees every hour the memory would summarise, so an explicit memory state adds
nothing. If that is the cause, then SHRINKING the window should make the memory the only
channel through which past information reaches the model, and the ablation gap should open.

Design: for lookback L in {1, 2, 3, 6, 12, 24}, train
    A1  pressure encoder, NO memory   (sees only the L-hour window)
    A2  pressure encoder + memory     (memory is computed over the same L hours)
    A2L pressure + LONG memory        (memory computed over a 48 h pressure history that
                                       the sequence encoder never sees -> the memory is
                                       genuinely the only long-range channel)

Decision rule, fixed BEFORE running:
  PASS if, at some L <= 6, A2/A2L beats A1 by >= 0.02 PR-AUC on the seed mean in at least
       4 of 5 test years, AND the learned half-life has seed std <= 1.5 h and mean within
       [2, 6] h (bracketing the measured 3.27 h episode half-life).
  FAIL otherwise -> the module goes to the appendix as a documented negative result.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.features import FEATURES_AR, FEATURES_CAL, build, load_hourly  # noqa: E402
from src.model import OPPN, PersistenceMemory, PressureEncoder  # noqa: E402

SEQ_FEATURES = ["neg_residual_load", "res_share", "solar_share", "wind_share",
                "res_fc", "load_fc", "resload_ramp_1h", "resload_ramp_3h"]
STATIC_FEATURES = FEATURES_CAL + FEATURES_AR + [
    "residual_load_daymin", "res_share_daymax", "resload_day_range"]
TEST_YEARS = [2021, 2022, 2023, 2024, 2025]
LOOKBACKS = [1, 2, 3, 6, 12, 24]
MEM_HISTORY = 48  # hours of pressure history available to the long-memory variant only


class OPPNLongMemory(nn.Module):
    """Short sequence window + a memory computed over a longer pressure history.

    The sequence encoder sees only the last L hours; the memory sees MEM_HISTORY hours.
    Any gain over A1 is therefore attributable to the memory channel alone.
    """

    def __init__(self, n_seq_features: int, n_static: int, hidden: int = 64,
                 pressure_hidden: int = 32, rho_init: float = 0.8, dropout: float = 0.1):
        super().__init__()
        self.seq_proj = nn.Sequential(nn.Linear(n_seq_features, hidden), nn.GELU(),
                                      nn.Dropout(dropout))
        self.pressure = PressureEncoder(n_seq_features, pressure_hidden)
        self.memory = PersistenceMemory(rho_init)
        pooled = hidden * 2 + n_static
        self.head = nn.Sequential(nn.Linear(pooled + 2, hidden), nn.GELU(),
                                  nn.Linear(hidden, 1))

    def forward(self, x_seq, x_static, x_hist):
        h = self.seq_proj(x_seq)
        pooled = torch.cat([h.mean(1), h[:, -1], x_static], dim=1)
        P_hist = self.pressure(x_hist)
        M = self.memory(P_hist)
        P_t = P_hist[:, -1]
        logit = self.head(torch.cat([pooled, P_t.unsqueeze(1), M.unsqueeze(1)], 1)).squeeze(-1)
        return {"logit": logit, "gate": torch.zeros_like(P_t), "P": P_t, "M": M}

    def diagnostics(self):
        return {"rho": float(self.memory.rho.detach()),
                "half_life_h": self.memory.half_life()}


def make_arrays(d: pd.DataFrame, target: str):
    d = d.copy()
    d["neg_residual_load"] = -d["residual_load"]
    seq = d[SEQ_FEATURES].to_numpy(np.float32)
    stat = d[STATIC_FEATURES].to_numpy(np.float32)
    y = d[target].to_numpy(np.float32)
    ok = np.isfinite(seq).all(1) & np.isfinite(stat).all(1) & np.isfinite(y)
    return seq, stat, y, ok, d.index


def windows(seq, stat, y, ok, index, lookback, hist_len):
    need = max(lookback, hist_len)
    rows = np.array([i for i in range(need - 1, len(y)) if ok[i - need + 1: i + 1].all()])
    X_seq = np.stack([seq[i - lookback + 1: i + 1] for i in rows])
    X_hist = np.stack([seq[i - hist_len + 1: i + 1] for i in rows])
    return X_seq, X_hist, stat[rows], y[rows], index[rows]


def standardize(train, *others):
    flat = train.reshape(-1, train.shape[-1])
    mu, sd = flat.mean(0), flat.std(0)
    sd[sd == 0] = 1.0
    return ((train - mu) / sd, *[(o - mu) / sd for o in others])


def train_eval(model, tensors, Y, masks, device, epochs, lr_rho=1e-2):
    tr, va, te = masks
    model = model.to(device)
    pos_w = torch.tensor([(len(Y[tr]) - Y[tr].sum()) / max(Y[tr].sum(), 1)],
                         device=device, dtype=torch.float32)
    lossfn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    rho_p = [p for n, p in model.named_parameters() if "memory" in n]
    other = [p for n, p in model.named_parameters() if "memory" not in n]
    groups = [{"params": other, "lr": 1e-3, "weight_decay": 1e-4}]
    if rho_p:
        groups.append({"params": rho_p, "lr": lr_rho, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups)

    train_t = [torch.tensor(t[tr]) for t in tensors] + [torch.tensor(Y[tr])]
    dl = DataLoader(TensorDataset(*train_t), batch_size=256, shuffle=True)
    va_t = [torch.tensor(t[va]).to(device) for t in tensors]
    te_t = [torch.tensor(t[te]).to(device) for t in tensors]

    best, state, bad = -np.inf, None, 0
    for _ in range(epochs):
        model.train()
        for *xb, yb in dl:
            out = model(*[x.to(device) for x in xb])
            loss = lossfn(out["logit"], yb.to(device))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(*va_t)["logit"]).cpu().numpy()
        s = average_precision_score(Y[va], pv)
        if s > best + 1e-5:
            best, bad = s, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 8:
                break
    if state:
        model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        pt = torch.sigmoid(model(*te_t)["logit"]).cpu().numpy()
    return float(average_precision_score(Y[te], pt)), model.diagnostics()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="y_run4")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    d, _ = build(load_hourly())
    seq, stat, y, ok, index = make_arrays(d, args.task)
    print(f"task={args.task} device={device} rows={len(y)}", flush=True)

    rows = []
    for L in LOOKBACKS:
        X_seq, X_hist, X_st, Y, idx = windows(seq, stat, y, ok, index, L, MEM_HISTORY)
        years = idx.year
        for ty in TEST_YEARS:
            tr, va, te = years < ty - 1, years == ty - 1, years == ty
            if tr.sum() < 2000 or va.sum() < 500 or te.sum() < 500:
                continue
            s_tr, s_va, s_te = standardize(X_seq[tr], X_seq[va], X_seq[te])
            h_tr, h_va, h_te = standardize(X_hist[tr], X_hist[va], X_hist[te])
            t_tr, t_va, t_te = standardize(X_st[tr], X_st[va], X_st[te])
            seq_all = np.empty_like(X_seq); seq_all[tr], seq_all[va], seq_all[te] = s_tr, s_va, s_te
            hist_all = np.empty_like(X_hist); hist_all[tr], hist_all[va], hist_all[te] = h_tr, h_va, h_te
            st_all = np.empty_like(X_st); st_all[tr], st_all[va], st_all[te] = t_tr, t_va, t_te
            masks = (tr, va, te)

            for seed in args.seeds:
                torch.manual_seed(seed); np.random.seed(seed)
                m1 = OPPN(X_seq.shape[-1], X_st.shape[-1], use_pressure=True,
                          use_persistence=False, use_gate=False)
                pr1, _ = train_eval(m1, [seq_all, st_all], Y, masks, device, args.epochs)

                torch.manual_seed(seed); np.random.seed(seed)
                m2 = OPPN(X_seq.shape[-1], X_st.shape[-1], use_pressure=True,
                          use_persistence=True, use_gate=False)
                pr2, dg2 = train_eval(m2, [seq_all, st_all], Y, masks, device, args.epochs)

                torch.manual_seed(seed); np.random.seed(seed)
                m3 = OPPNLongMemory(X_seq.shape[-1], X_st.shape[-1])
                pr3, dg3 = train_eval(m3, [seq_all, st_all, hist_all], Y, masks, device,
                                      args.epochs)

                rows.append({"lookback": L, "test_year": ty, "seed": seed,
                             "A1_no_memory": pr1, "A2_memory": pr2,
                             "A2L_long_memory": pr3,
                             "hl_A2": dg2.get("half_life_h"),
                             "hl_A2L": dg3.get("half_life_h")})
                print(f"  L={L} {ty} s{seed}: A1={pr1:.3f} A2={pr2:.3f} A2L={pr3:.3f} "
                      f"HL_A2={dg2.get('half_life_h'):.2f} HL_A2L={dg3.get('half_life_h'):.2f}",
                      flush=True)

    res = pd.DataFrame(rows)
    out = PROJECT_ROOT / "outputs" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / f"lookback_diagnosis_{args.task}.csv", index=False)

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    agg = res.groupby(["lookback", "test_year"]).mean(numeric_only=True).reset_index()
    print("\n=== PR-AUC gain of memory over no-memory (seed mean) ===")
    agg["gain_A2"] = agg.A2_memory - agg.A1_no_memory
    agg["gain_A2L"] = agg.A2L_long_memory - agg.A1_no_memory
    print(agg.pivot(index="lookback", columns="test_year", values="gain_A2").round(3).to_string())
    print("\n=== PR-AUC gain of LONG memory (only long-range channel) ===")
    print(agg.pivot(index="lookback", columns="test_year", values="gain_A2L").round(3).to_string())
    print("\n=== Learned half-life stability by lookback ===")
    print(res.groupby("lookback")[["hl_A2", "hl_A2L"]]
          .agg(["mean", "std"]).round(2).to_string())


if __name__ == "__main__":
    sys.exit(main())
