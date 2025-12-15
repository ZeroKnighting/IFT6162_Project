#!/usr/bin/env python3
"""
lao_mpc_controller.py

Drop-in Learning-Augmented Optimization (LAO) controller for your PondCSTR MPC code.

What it does
------------
- Loads a trained checkpoint (PolicyNet + normalization stats; optionally ValueNet too).
- Uses PolicyNet to predict (theta0, theta1) from current state x=[h,c].
- Prunes the grid-search candidates around the predicted thetas (k-nearest on the candidate grid).
- Runs the same CH=2 move-blocking MPC as your run_deterministic_mpc, but faster.

How to use (recommended)
------------------------
In your existing benchmark script (the one that defines PondCSTR etc.):

    from lao_mpc_controller import run_lao_pruned_mpc

    traj_lao = run_lao_pruned_mpc(
        model=model,
        x0=x0,
        u0=u0,
        qin_forecast=qin_f,
        cin_forecast=cin_f,
        MD_exec_true=MD_t,
        horizon=horizon,
        theta_candidates=theta_candidates,
        hlimit_ft=hlimit_ft,
        ckpt_path="lao_models.pt",
        k_near=5,
    )

Then include traj_lao in your trajs dict for compute_metrics().

Notes
-----
- This file is intentionally self-contained except it expects your model object provides:
    model.area_vec(h_array) and model.q_out_vec(h_array, theta_array)
  which your PondCSTR already does.
- PolicyNet architecture assumed: 2 -> hidden -> 2 with sigmoid output.
  The checkpoint should include 'policy_state_dict', and 'x_mean', 'x_std'.

Optional extension
------------------
- You can later add terminal value (ValueNet) into planning; for now this is "pruning-only"
  which is the safest first LAO step.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Callable

import numpy as np
import time

# torch is only needed for the policy network inference
import torch
import torch.nn as nn

from train_lao_nets import PolicyNet

@dataclass
class LAOPolicyPack:
    policy: PolicyNet
    x_mean: np.ndarray  # (1,2)
    x_std: np.ndarray   # (1,2)
    device: str


def load_lao_policy(ckpt_path: str, hidden: int = 64, device: Optional[str] = None) -> LAOPolicyPack:
    """
    Load policy network + input normalization stats from a checkpoint saved by your trainer.
    Expected keys:
        - policy_state_dict
        - x_mean
        - x_std
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(ckpt_path, map_location=device)

    # tolerate a few naming variants
    sd_key = "policy_state_dict" if "policy_state_dict" in ckpt else "policy"
    if sd_key not in ckpt:
        raise KeyError(f"Checkpoint missing policy weights. Found keys: {list(ckpt.keys())}")

    x_mean = np.asarray(ckpt.get("x_mean", ckpt.get("X_mean", None)), dtype=np.float64)
    x_std  = np.asarray(ckpt.get("x_std",  ckpt.get("X_std",  None)), dtype=np.float64)
    if x_mean is None or x_std is None:
        raise KeyError("Checkpoint missing x_mean/x_std for input normalization.")

    # ensure shape (1,2)
    x_mean = x_mean.reshape(1, -1)
    x_std = x_std.reshape(1, -1)

    policy = PolicyNet(in_dim=x_mean.shape[1], hidden=hidden).to(device)
    policy.load_state_dict(ckpt[sd_key])
    policy.eval()

    return LAOPolicyPack(policy=policy, x_mean=x_mean, x_std=x_std, device=device)


# ----------------------------
# Pruning helpers
# ----------------------------
def _k_nearest_candidates(pred: float, grid: np.ndarray, k_near: int) -> np.ndarray:
    """
    Return k_near candidate values from grid that are closest to pred.
    """
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
    """
    Like your plan_theta_pair_grid_fast, but allows separate candidate sets for theta0 and theta1.
    Returns best (theta0, theta1, best_cost).
    """
    cand0 = np.asarray(cand0, dtype=float)
    cand1 = np.asarray(cand1, dtype=float)

    th0_grid, th1_grid = np.meshgrid(cand0, cand1, indexing="ij")  # (n0,n1)
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
    hidden: int = 128,
    k_near: int = 5,
    fallback_full_grid: bool = True,
) -> ControllerTraj:
    """
    Deterministic MPC-false execution, but planner grid is pruned around PolicyNet prediction.

    Parameters
    ----------
    k_near : number of nearest candidates (per dimension) kept around predicted theta.
             effective pair evaluations = k_near^2 (e.g., 25 if k_near=5)
    fallback_full_grid : if pruned grid yields inf (no feasible), fall back to full grid that step.

    Returns
    -------
    ControllerTraj like your existing functions.
    """
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

    for k in range(Tsim - horizon - 1):
        xk = np.array([h[k], c[k]], dtype=float)

        # policy inference: predict (theta0, theta1)
        xn = (xk.reshape(1, -1) - pack.x_mean) / pack.x_std
        xt = torch.tensor(xn, dtype=torch.float32, device=pack.device)
        with torch.no_grad():
            pred = pack.policy(xt).detach().cpu().numpy().reshape(-1)
        pred_th0 = float(np.clip(pred[0], 0.0, 1.0))
        pred_th1 = float(np.clip(pred[1], 0.0, 1.0))

        # prune candidates
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

        # if pruned grid fails (infeasible), optionally fall back to full grid
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

        # optional: store warm-start second move
        theta[k + 1] = th1

    # fill trailing qout if needed (same pattern as your baseline)
    for k in range(Tsim - 1, -1, -1):
        if qout[k] == 0.0 and k > 0:
            u_real = np.array([theta[k], MD_exec_true[k, 0], MD_exec_true[k, 1]], dtype=float)
            qout[k] = model.output(np.array([h[k], c[k]], dtype=float), u_real)[2]

    elapsed_time = time.perf_counter() - t0
    print(f"LAO-pruned MPC done in {elapsed_time:.2f}s")

    return ControllerTraj(h_ft=h, c=c, qout_cfs=qout, theta=theta, qspill_cfs=None, elapsed_time=elapsed_time)
