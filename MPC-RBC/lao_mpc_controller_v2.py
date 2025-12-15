#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import time

import torch
import torch.nn as nn


# ----------------------------
# Local PolicyNet (avoid external import dependency)
# ----------------------------
class PolicyNet(nn.Module):
    """2-layer MLP: Linear->ReLU->Linear->Sigmoid, outputs (theta0, theta1) in (0,1)."""
    def __init__(self, in_dim: int = 2, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class LAOPolicyPack:
    policy: PolicyNet
    x_mean: np.ndarray  # (1,2)
    x_std: np.ndarray   # (1,2)
    device: str


def _get_ckpt_key(ckpt: dict, candidates: Tuple[str, ...]) -> Optional[str]:
    for k in candidates:
        if k in ckpt:
            return k
    return None


def load_lao_policy(ckpt_path: str, hidden: int = 64, device: Optional[str] = None) -> LAOPolicyPack:
    """
    Load policy network + input normalization stats from a checkpoint.

    Expected keys (flexible):
      - policy_state_dict OR policy OR (rare) "policy_net"
      - x_mean/x_std (or X_mean/X_std)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(ckpt_path, map_location=device)

    # ---- policy weights key ----
    sd_key = _get_ckpt_key(ckpt, ("policy_state_dict", "policy", "policy_net"))
    if sd_key is None:
        raise KeyError(f"Checkpoint missing policy weights. Found keys: {list(ckpt.keys())}")

    # ---- normalization stats (robust) ----
    xm_key = _get_ckpt_key(ckpt, ("x_mean", "X_mean"))
    xs_key = _get_ckpt_key(ckpt, ("x_std", "X_std"))
    if xm_key is None or xs_key is None:
        raise KeyError("Checkpoint missing x_mean/x_std (or X_mean/X_std) for input normalization.")

    x_mean = ckpt[xm_key]
    x_std = ckpt[xs_key]

    # Convert safely
    x_mean = np.asarray(x_mean, dtype=np.float64).reshape(1, -1)
    x_std  = np.asarray(x_std,  dtype=np.float64).reshape(1, -1)

    if x_mean.shape[1] != 2:
        # still allow, but warn via error message (PondCSTR state is 2-dim)
        raise ValueError(f"x_mean expected shape (1,2), got {x_mean.shape}")
    if x_std.shape != x_mean.shape:
        raise ValueError(f"x_std shape must match x_mean. Got x_std={x_std.shape}, x_mean={x_mean.shape}")

    # Avoid divide-by-zero
    x_std = np.maximum(x_std, 1e-8)

    if ("large" in ckpt_path.lower()) or ("real_case" in ckpt_path.lower()):
        from train_lao_nets import PolicyNet_real, ValueNet_real
        policy = PolicyNet_real(in_dim=x_mean.shape[1], hidden=hidden).to(device)
    else:
        policy = PolicyNet(in_dim=x_mean.shape[1], hidden=hidden).to(device)
    # policy = PolicyNet(in_dim=x_mean.shape[1], hidden=hidden).to(device)
    policy.load_state_dict(ckpt[sd_key])
    policy.eval()

    return LAOPolicyPack(policy=policy, x_mean=x_mean, x_std=x_std, device=device)


# ----------------------------
# Pruning helper
# ----------------------------
def _k_nearest_candidates(pred: float, grid: np.ndarray, k_near: int) -> np.ndarray:
    grid = np.asarray(grid, dtype=float)
    k = int(max(1, min(k_near, grid.size)))
    idx = np.argsort(np.abs(grid - float(pred)))[:k]
    return np.sort(grid[idx])


# ----------------------------
# Fast planner over cand0 x cand1 with best cost (CH=2)
# ----------------------------
def plan_theta_pair_grid_fast_with_cost_pruned(
    model,
    xk: np.ndarray,
    q_forecast: np.ndarray,
    c_forecast: np.ndarray,
    cand0: np.ndarray,
    cand1: np.ndarray,
    hlimit_ft: float,
) -> Tuple[float, float, float]:
    cand0 = np.asarray(cand0, dtype=float)
    cand1 = np.asarray(cand1, dtype=float)

    th0_grid, th1_grid = np.meshgrid(cand0, cand1, indexing="ij")
    th0 = th0_grid.reshape(-1)
    th1 = th1_grid.reshape(-1)
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
    cost[violated] = np.inf

    best = int(np.argmin(cost))
    return float(th0[best]), float(th1[best]), float(cost[best])


# ----------------------------
# Main LAO MPC runner (pruning-only)
# ----------------------------
@dataclass
class ControllerTraj:
    h_ft: np.ndarray
    c: np.ndarray
    qout_cfs: np.ndarray
    theta: np.ndarray
    qspill_cfs: Optional[np.ndarray] = None
    elapsed_time: Optional[float] = None


def run_lao_pruned_mpc(
    model,
    x0: np.ndarray,
    u0: float,
    qin_forecast: np.ndarray,
    cin_forecast: np.ndarray,
    MD_exec_true: np.ndarray,
    horizon: int,
    theta_candidates: np.ndarray,
    hlimit_ft: float,
    ckpt_path: str,
    hidden: int = 64,
    k_near: int = 5,
    fallback_full_grid: bool = True,
    torch_threads: Optional[int] = 1,
) -> ControllerTraj:
    """
    Deterministic MPC execution with LAO pruning using PolicyNet.

    torch_threads: set to 1 to reduce overhead (often faster for tiny nets).
    """
    if torch_threads is not None:
        try:
            torch.set_num_threads(int(torch_threads))
        except Exception:
            pass

    pack = load_lao_policy(ckpt_path=ckpt_path, hidden=hidden)

    t0 = time.perf_counter()

    Tsim = len(qin_forecast)
    h = np.zeros(Tsim, dtype=float)
    c = np.zeros(Tsim, dtype=float)
    qout = np.zeros(Tsim, dtype=float)
    theta = np.zeros(Tsim, dtype=float)

    h[0], c[0] = float(x0[0]), float(x0[1])
    theta[0] = float(np.clip(u0, 0.0, 1.0))

    cand_full = np.asarray(theta_candidates, dtype=float)

    # Use a no_grad context for the whole loop (less overhead)
    with torch.no_grad():
        for k in range(Tsim - horizon - 1):
            xk = np.array([h[k], c[k]], dtype=float)

            # policy inference (numpy -> torch with minimal overhead)
            xn = (xk.reshape(1, -1) - pack.x_mean) / pack.x_std
            xt = torch.from_numpy(xn.astype(np.float32, copy=False)).to(pack.device)

            pred = pack.policy(xt).cpu().numpy().reshape(-1)
            pred_th0 = float(np.clip(pred[0], 0.0, 1.0))
            pred_th1 = float(np.clip(pred[1], 0.0, 1.0))

            cand0 = _k_nearest_candidates(pred_th0, cand_full, k_near=k_near)
            cand1 = _k_nearest_candidates(pred_th1, cand_full, k_near=k_near)

            q_base = np.asarray(qin_forecast[k : k + horizon], dtype=float)
            c_base = np.asarray(cin_forecast[k : k + horizon], dtype=float)

            th0, th1, Jbest = plan_theta_pair_grid_fast_with_cost_pruned(
                model=model,
                xk=xk,
                q_forecast=q_base,
                c_forecast=c_base,
                cand0=cand0,
                cand1=cand1,
                hlimit_ft=hlimit_ft,
            )

            if (not np.isfinite(Jbest)) and fallback_full_grid:
                th0, th1, Jbest = plan_theta_pair_grid_fast_with_cost_pruned(
                    model=model,
                    xk=xk,
                    q_forecast=q_base,
                    c_forecast=c_base,
                    cand0=cand_full,
                    cand1=cand_full,
                    hlimit_ft=hlimit_ft,
                )

            theta_apply = th0
            theta[k] = theta_apply

            u_real = np.array([theta_apply, MD_exec_true[k, 0], MD_exec_true[k, 1]], dtype=float)
            qout[k] = model.output(xk, u_real)[2]

            x_next = model.state_step(xk, u_real)
            h[k + 1], c[k + 1] = float(x_next[0]), float(x_next[1])

            # optional warm-start storage
            theta[k + 1] = th1

    for k in range(Tsim - 1, -1, -1):
        if qout[k] == 0.0 and k > 0:
            u_real = np.array([theta[k], MD_exec_true[k, 0], MD_exec_true[k, 1]], dtype=float)
            qout[k] = model.output(np.array([h[k], c[k]], dtype=float), u_real)[2]

    elapsed_time = time.perf_counter() - t0
    print(f"LAO-pruned MPC done in {elapsed_time:.2f}s")

    return ControllerTraj(h_ft=h, c=c, qout_cfs=qout, theta=theta, qspill_cfs=None, elapsed_time=elapsed_time)
