
"""
lao_terminal_value_mpc.py

Learning-Augmented MPC using a learned terminal value function V(x_H).
- Uses your CH=2 move-blocking grid search planner (theta0, theta1).
- Adds terminal cost: J = stage_cost + terminal_weight * V(x_H)
- Intended to let you shorten the MPC horizon (e.g., 96 -> 24/32/48) while
  retaining long-horizon behavior.

This module is designed to drop into your existing project:
    from lao_terminal_value_mpc import run_lao_terminal_value_mpc

Requirements:
    pip install torch
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import time

try:
    import torch
    import torch.nn as nn
except Exception as e:  # pragma: no cover
    raise ImportError("PyTorch is required for lao_terminal_value_mpc.py. Install with `pip install torch`.") from e


# -----------------------------
# Data container (match your code)
# -----------------------------
@dataclass
class ControllerTraj:
    h_ft: np.ndarray
    c: np.ndarray
    qout_cfs: np.ndarray
    theta: np.ndarray
    qspill_cfs: Optional[np.ndarray] = None
    elapsed_time: Optional[float] = None


# -----------------------------
# Value network definition
# (2-layer MLP: Linear->ReLU->Linear)
# -----------------------------
class ValueNet(nn.Module):
    def __init__(self, in_dim: int = 2, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def _infer_hidden_from_state_dict(state_dict: dict) -> int:
    # Works for either keys like "net.0.weight" or "fc1.weight"
    for k, v in state_dict.items():
        if k.endswith("fc1.weight") and v.ndim == 2:
            return int(v.shape[0])
    # fallback: look for first Linear weight
    for k, v in state_dict.items():
        if k.endswith("weight") and v.ndim == 2 and v.shape[1] == 2:
            return int(v.shape[0])
    raise ValueError("Could not infer hidden size from checkpoint state_dict.")


def load_value_terminal_fn(
    ckpt_path: str,
    device: Optional[str] = None,
) -> Tuple[Callable[[np.ndarray], float], dict]:
    """
    Loads ValueNet and returns a callable V(x_np)->float in RAW J* space.
    Supports checkpoints that store value target transform params:
        - v_mean, v_std, value_transform="log1p+zscore" (recommended)
    Also supports x normalization params:
        - x_mean, x_std

    Returns:
        V_terminal_fn, info_dict
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(ckpt_path, map_location=device)

    # Be tolerant to different checkpoint formats
    if "value_state_dict" in ckpt:
        v_sd = ckpt["value_state_dict"]
    elif "value_net" in ckpt and isinstance(ckpt["value_net"], dict):
        v_sd = ckpt["value_net"]
    elif "state_dict" in ckpt:
        v_sd = ckpt["state_dict"]
    else:
        # maybe it's a bare state_dict
        v_sd = ckpt

    hidden = ckpt.get("hidden", None)
    if hidden is None:
        hidden = _infer_hidden_from_state_dict(v_sd)

    vnet = ValueNet(in_dim=2, hidden=int(hidden)).to(device)
    vnet.load_state_dict(v_sd, strict=False)
    vnet.eval()

    # Input normalization
    x_mean = np.asarray(ckpt.get("x_mean", np.zeros((1, 2), dtype=float)), dtype=float)
    x_std = np.asarray(ckpt.get("x_std", np.ones((1, 2), dtype=float)), dtype=float)

    # Value transform (log1p+zscore)
    value_transform = ckpt.get("value_transform", None)
    v_mean = ckpt.get("v_mean", None)
    v_std = ckpt.get("v_std", None)

    if v_mean is not None:
        v_mean = np.asarray(v_mean, dtype=float).reshape(1, 1)
    if v_std is not None:
        v_std = np.asarray(v_std, dtype=float).reshape(1, 1)

    @torch.no_grad()
    def V_raw(x_np: np.ndarray) -> float:
        x_np = np.asarray(x_np, dtype=float).reshape(1, 2)
        x_n = (x_np - x_mean) / (x_std + 1e-12)
        xt = torch.tensor(x_n, dtype=torch.float32, device=device)
        z = vnet(xt).cpu().numpy().reshape(1, 1)  # network output

        # If trained on standardized log1p(J), invert to raw J
        if value_transform == "log1p+zscore" and v_mean is not None and v_std is not None:
            j_log = z * v_std + v_mean
            j = np.expm1(j_log)  # inverse log1p
            return float(j.reshape(()))

        # Otherwise, assume network outputs raw J directly
        return float(z.reshape(()))

    info = {
        "device": device,
        "hidden": int(hidden),
        "value_transform": value_transform,
        "has_x_norm": True,
        "has_value_norm": (value_transform == "log1p+zscore" and v_mean is not None and v_std is not None),
    }
    return V_raw, info


# -----------------------------
# Planner with terminal cost
# -----------------------------
def plan_theta_pair_grid_fast_with_cost(
    model,
    xk: np.ndarray,
    q_forecast: np.ndarray,
    c_forecast: np.ndarray,
    theta_candidates: np.ndarray,
    hlimit_ft: float,
    terminal_value_fn: Optional[Callable[[np.ndarray], float]] = None,
    terminal_weight: float = 1.0,
) -> Tuple[float, float, float]:
    """
    Vectorized grid search for (theta0, theta1) with CH=2 move-blocking.
    Returns (best_theta0, best_theta1, best_cost) where best_cost includes
    terminal_weight * V(x_H) if terminal_value_fn is provided.
    """
    cand = np.asarray(theta_candidates, dtype=float)
    th0_grid, th1_grid = np.meshgrid(cand, cand, indexing="ij")
    th0 = th0_grid.reshape(-1)  # (P,)
    th1 = th1_grid.reshape(-1)  # (P,)
    P = th0.size

    H = int(len(q_forecast))
    qf = np.asarray(q_forecast, dtype=float)
    cf = np.asarray(c_forecast, dtype=float)

    h = np.full(P, float(xk[0]), dtype=float)
    c = np.full(P, float(xk[1]), dtype=float)

    violated = np.zeros(P, dtype=bool)

    sum_cq2 = np.zeros(P, dtype=float)
    sum_q = np.zeros(P, dtype=float)
    sum_q2 = np.zeros(P, dtype=float)

    exp_kdt = float(np.exp(-model.k * model.dt))

    for j in range(H):
        theta = th0 if j == 0 else th1
        A = model.area_vec(h)
        qout = model.q_out_vec(h, theta)

        cq = c * qout
        sum_cq2 += cq * cq
        sum_q += qout
        sum_q2 += qout * qout

        q_in = qf[j]
        c_in = cf[j]

        h_next = np.maximum(0.0, h + (model.dt / A) * (q_in - qout))

        den = np.maximum((A * h) + q_in * model.dt, 1e-6)
        c_next = np.where(
            h > 0.0,
            (c * A * h * exp_kdt + c_in * q_in * model.dt) / den,
            0.0,
        )

        violated |= (h_next > hlimit_ft)
        h, c = h_next, c_next

    qbar = sum_q / max(H, 1)
    smooth = sum_q2 - H * (qbar * qbar)
    cost = 5.0 * sum_cq2 + smooth + 900.0 * (h * h)

    if terminal_value_fn is not None:
        V = np.empty(P, dtype=float)
        for i in range(P):
            V[i] = float(terminal_value_fn(np.array([h[i], c[i]], dtype=float)))
        cost = cost + float(terminal_weight) * V

    cost[violated] = np.inf
    best = int(np.argmin(cost))
    return float(th0[best]), float(th1[best]), float(cost[best])


# -----------------------------
# Controller: shorter-horizon MPC + learned terminal value
# -----------------------------
def run_lao_terminal_value_mpc(
    model,
    x0: np.ndarray,
    u0: float,
    qin_forecast: np.ndarray,
    cin_forecast: np.ndarray,
    MD_exec_true: np.ndarray,
    horizon: int,
    theta_candidates: np.ndarray,
    hlimit_ft: float,
    ckpt_path: str = "lao_models.pt",
    terminal_weight: float = 1.0,
    device: Optional[str] = None,
) -> ControllerTraj:
    """
    Deterministic MPC with CH=2 move-blocking + learned terminal value V(x_H).

    Typical use:
        - horizon = 24/32/48 (smaller than the baseline 96)
        - terminal_weight tuned around 0.1~5.0 (start with 1.0)

    Planning uses (qin_forecast, cin_forecast).
    Execution uses true disturbances MD_exec_true = [qin_t, cin_t].
    """
    V_term, info = load_value_terminal_fn(ckpt_path=ckpt_path, device=device)

    t0 = time.perf_counter()

    Tsim = len(qin_forecast)
    h = np.zeros(Tsim, dtype=float)
    c = np.zeros(Tsim, dtype=float)
    qout = np.zeros(Tsim, dtype=float)
    theta = np.zeros(Tsim, dtype=float)

    h[0], c[0] = float(x0[0]), float(x0[1])
    theta[0] = float(np.clip(u0, 0.0, 1.0))

    warm = (theta[0], theta[0])

    for k in range(Tsim - horizon - 1):
        q_base = qin_forecast[k : k + horizon]
        c_base = cin_forecast[k : k + horizon]

        th0, th1, _ = plan_theta_pair_grid_fast_with_cost(
            model=model,
            xk=np.array([h[k], c[k]], dtype=float),
            q_forecast=q_base,
            c_forecast=c_base,
            theta_candidates=theta_candidates,
            hlimit_ft=hlimit_ft,
            terminal_value_fn=V_term,
            terminal_weight=terminal_weight,
        )

        theta_apply = th0
        theta[k] = theta_apply

        u_real = np.array([theta_apply, MD_exec_true[k, 0], MD_exec_true[k, 1]], dtype=float)
        qout[k] = model.output(np.array([h[k], c[k]], dtype=float), u_real)[2]

        x_next = model.state_step(np.array([h[k], c[k]], dtype=float), u_real)
        h[k + 1], c[k + 1] = float(x_next[0]), float(x_next[1])

        warm = (th1, th1)
        theta[k + 1] = th1

    for k in range(Tsim - 1, -1, -1):
        if qout[k] == 0.0 and k > 0:
            u_real = np.array([theta[k], MD_exec_true[k, 0], MD_exec_true[k, 1]], dtype=float)
            qout[k] = model.output(np.array([h[k], c[k]], dtype=float), u_real)[2]

    elapsed_time = time.perf_counter() - t0
    print(f"LAO-terminal-value MPC done in {elapsed_time:.2f}s")

    return ControllerTraj(h_ft=h, c=c, qout_cfs=qout, theta=theta, qspill_cfs=None, elapsed_time=elapsed_time)