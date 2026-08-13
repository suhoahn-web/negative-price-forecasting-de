"""Oversupply-Pressure Persistence Network (OPPN).

Architecture (deliberately small; the contribution is the estimated timescale, not the net):

    day-ahead forecast window  ->  Pressure encoder  ->  P_t >= 0  (oversupply pressure)
                                            |
                                            v
                         Learned persistence memory  M_t = sum_k (1-rho) rho^k P_{t-k}
                                            |
                          +-----------------+------------------+
                          |                                    |
                     Soft gate g_t                    Oversupply head
                          |                                    |
              (1-g) * normal head  +  g * oversupply head  ->  logit P(y=1)

rho = sigmoid(theta) is learned end-to-end; implied half-life = ln(0.5)/ln(rho) HOURS.
The scientific test: does the learned half-life recover the measured negative-price
episode half-life (~3.3 h) rather than the price-level half-life (~48 h)?

Honest framing: M_t is exponential smoothing with a learned coefficient (Smyl 2020;
ETSformer; GRU-D; Tallec & Ollivier 2018 show a learnable gate IS a learnable timescale).
The novelty is estimating and validating a market timescale, not the mechanism.
"""

import math

import torch
import torch.nn as nn


class PressureEncoder(nn.Module):
    """Maps day-ahead forecast features at each hour to a non-negative pressure scalar.

    A direct linear skip on the (standardized) negative residual load anchors the signal
    to an observable quantity so it cannot collapse to a constant at initialization.
    """

    def __init__(self, n_features: int, hidden: int = 32, skip_index: int = 0):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(n_features, hidden), nn.GELU(),
                                 nn.Linear(hidden, 1))
        self.skip_w = nn.Parameter(torch.tensor(1.0))
        self.skip_index = skip_index
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, F) -> (B, L) non-negative pressure."""
        raw = self.mlp(x).squeeze(-1) + self.skip_w * x[..., self.skip_index]
        return self.softplus(raw)


class PersistenceMemory(nn.Module):
    """M_t = sum_{k=0}^{L-1} (1-rho) rho^k P_{t-k}, rho = sigmoid(theta), learned."""

    def __init__(self, rho_init: float = 0.8):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(math.log(rho_init / (1 - rho_init))))

    @property
    def rho(self) -> torch.Tensor:
        return torch.sigmoid(self.theta)

    def half_life(self) -> float:
        r = float(self.rho.detach())
        return math.log(0.5) / math.log(r) if 0 < r < 1 else float("nan")

    def forward(self, P: torch.Tensor) -> torch.Tensor:
        """P: (B, L) oldest -> newest. Returns M: (B,)."""
        L = P.size(1)
        rho = self.rho
        k = torch.arange(L, device=P.device, dtype=P.dtype)
        w = torch.flip((1 - rho) * rho ** k, dims=[0])  # align newest-last
        return (P * w).sum(dim=1)


class OPPN(nn.Module):
    def __init__(self, n_seq_features: int, n_static: int,
                 hidden: int = 64, pressure_hidden: int = 32, gate_hidden: int = 16,
                 rho_init: float = 0.8, dropout: float = 0.1,
                 use_persistence: bool = True, use_gate: bool = True,
                 use_pressure: bool = True):
        super().__init__()
        self.use_persistence = use_persistence
        self.use_gate = use_gate
        self.use_pressure = use_pressure

        # shared encoder over the forecast window (mean+last pooling, no attention needed
        # at this data size)
        self.seq_proj = nn.Sequential(nn.Linear(n_seq_features, hidden), nn.GELU(),
                                      nn.Dropout(dropout))
        pooled = hidden * 2 + n_static

        if use_pressure:
            self.pressure = PressureEncoder(n_seq_features, pressure_hidden)
        if use_persistence:
            self.memory = PersistenceMemory(rho_init)

        extra = (1 if use_pressure else 0) + (1 if use_persistence else 0)
        if use_gate:
            self.gate = nn.Sequential(nn.Linear(extra + n_static, gate_hidden), nn.GELU(),
                                      nn.Linear(gate_hidden, 1))
            nn.init.constant_(self.gate[-1].bias, -1.0)  # start on the normal head
            self.head_normal = nn.Linear(pooled, 1)
            self.head_shock = nn.Sequential(nn.Linear(pooled + extra, hidden), nn.GELU(),
                                            nn.Linear(hidden, 1))
        else:
            self.head = nn.Sequential(nn.Linear(pooled + extra, hidden), nn.GELU(),
                                      nn.Linear(hidden, 1))

    def forward(self, x_seq: torch.Tensor, x_static: torch.Tensor) -> dict:
        B = x_seq.size(0)
        dev = x_seq.device
        h = self.seq_proj(x_seq)
        pooled = torch.cat([h.mean(dim=1), h[:, -1], x_static], dim=1)

        P_t = torch.zeros(B, device=dev)
        M_t = torch.zeros(B, device=dev)
        extras = []
        if self.use_pressure:
            P_seq = self.pressure(x_seq)
            P_t = P_seq[:, -1]
            extras.append(P_t.unsqueeze(1))
            if self.use_persistence:
                M_t = self.memory(P_seq)
                extras.append(M_t.unsqueeze(1))
        elif self.use_persistence:
            # persistence over the raw anchor feature when the encoder is ablated out
            M_t = self.memory(x_seq[..., 0])
            extras.append(M_t.unsqueeze(1))
        extra = torch.cat(extras, dim=1) if extras else torch.zeros(B, 0, device=dev)

        if self.use_gate:
            g = torch.sigmoid(self.gate(torch.cat([extra, x_static], dim=1))).squeeze(-1)
            logit = ((1 - g) * self.head_normal(pooled).squeeze(-1)
                     + g * self.head_shock(torch.cat([pooled, extra], dim=1)).squeeze(-1))
            return {"logit": logit, "gate": g, "P": P_t, "M": M_t}
        logit = self.head(torch.cat([pooled, extra], dim=1)).squeeze(-1)
        return {"logit": logit, "gate": torch.zeros(B, device=dev), "P": P_t, "M": M_t}

    def diagnostics(self) -> dict:
        if self.use_persistence:
            return {"rho": float(self.memory.rho.detach()),
                    "half_life_h": self.memory.half_life()}
        return {}
