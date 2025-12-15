from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

try:
    from scipy.io import savemat  # type: ignore
except Exception:
    savemat = None

try:
    from scipy.interpolate import CubicSpline  # type: ignore
except Exception:
    CubicSpline = None

from typing import Tuple
import numpy as np


# =============================================================================
# Data containers
# =============================================================================

@dataclass
class PassiveResult:
    Xopt: np.ndarray  # (T,2) [h,c]
    Yopt: np.ndarray  # (T,3) [h,c,q_out]


@dataclass
class ControllerTraj:
    h_ft: np.ndarray
    c: np.ndarray
    qout_cfs: np.ndarray
    theta: np.ndarray
    qspill_cfs: Optional[np.ndarray] = None  # overflow flow (cfs), if modeled


# =============================================================================
# Pond CSTR model
# =============================================================================

class PondCSTR:
    """
    Discrete-time dynamics (15 min) aligned to pondcstr_StateFcn / OutputFcn.
    Units:
      - h in ft
      - A(h) in ft^2
      - q in ft^3/s (cfs)
      - g in ft/s^2
      - dt in s
      - c is arbitrary concentration unit (as in input)
    """

    def __init__(
        self,
        dt: float = 15.0 * 60.0,   # 900 s
        co: float = 0.65,
        Ao: float = 1.0,
        g: float = 32.2,
        k_day: float = 0.8,        # 0.8 / day
        elevation: Optional[np.ndarray] = None,
        area: Optional[np.ndarray] = None,
        use_spline: bool = True,
    ):
        self.dt = float(dt)
        self.co = float(co)
        self.Ao = float(Ao)
        self.g = float(g)
        self.k = float(k_day) / 24.0 / 60.0 / 60.0  # s^-1

        if elevation is None:
            elevation = np.array([0, 2, 4, 6, 8, 10], dtype=float)
        if area is None:
            area = np.array([82971, 93258, 106100, 119152, 134285, 134285], dtype=float)

        self.elevation = np.asarray(elevation, dtype=float)
        self.area_tab = np.asarray(area, dtype=float)

        self._area_spline = None
        if use_spline and CubicSpline is not None:
            # "spline" close to MATLAB interp1(...,'spline')
            self._area_spline = CubicSpline(self.elevation, self.area_tab, bc_type="natural")

    def area(self, h_ft: float) -> float:
        h = float(max(0.0, h_ft))
        if self._area_spline is not None:
            # clamp domain to avoid spline extrapolation weirdness
            h_clamped = float(np.clip(h, self.elevation.min(), self.elevation.max()))
            A = float(self._area_spline(h_clamped))
            return max(A, 1e-6)
        # fallback: linear
        A = float(np.interp(h, self.elevation, self.area_tab))
        return max(A, 1e-6)

    def q_out(self, h_ft: float, theta: float) -> float:
        """
        q_out = theta*co*Ao*sqrt(2*g*h)*min(1,h)
        """
        h = float(max(0.0, h_ft))
        th = float(np.clip(theta, 0.0, 1.0))
        return th * self.co * self.Ao * np.sqrt(2.0 * self.g * h) * min(1.0, h)

    def state_step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        x = [h, c]
        u = [theta, q_in, c_in]
        """
        h, c = float(x[0]), float(x[1])
        theta, q_in, c_in = float(u[0]), float(u[1]), float(u[2])

        h = max(0.0, h)
        A = self.area(h)
        qout = self.q_out(h, theta)

        next_h = max(0.0, h + self.dt / A * (q_in - qout))
        if h > 0.0:
            num = (c * A * h * np.exp(-self.k * self.dt)) + c_in * q_in * self.dt
            den = (A * h) + q_in * self.dt
            next_c = num / max(den, 1e-6)
        else:
            next_c = 0.0

        return np.array([next_h, next_c], dtype=float)

    def output(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        y = [h, c, q_out] using current x and u.
        """
        h, c = float(x[0]), float(x[1])
        theta = float(u[0])
        qout = self.q_out(h, theta)
        return np.array([max(0.0, h), max(0.0, c), qout], dtype=float)

    def area_vec(self, h_ft: np.ndarray) -> np.ndarray:
        h = np.maximum(0.0, np.asarray(h_ft, dtype=float))
        if self._area_spline is not None:
            h = np.clip(h, self.elevation.min(), self.elevation.max())
            A = self._area_spline(h)
        else:
            A = np.interp(h, self.elevation, self.area_tab)
        return np.maximum(A, 1e-6)

    def q_out_vec(self, h_ft: np.ndarray, theta: np.ndarray) -> np.ndarray:
        h = np.maximum(0.0, np.asarray(h_ft, dtype=float))
        th = np.clip(np.asarray(theta, dtype=float), 0.0, 1.0)
        return th * self.co * self.Ao * np.sqrt(2.0 * self.g * h) * np.minimum(1.0, h)


# =============================================================================
# Simple EKF (numerical Jacobians) for MPC-EKF
# measurement: z = c + noise
# =============================================================================

def _num_jacobian(f, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y0 = np.asarray(f(x), dtype=float)
    J = np.zeros((y0.size, x.size), dtype=float)
    for i in range(x.size):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        yp = np.asarray(f(xp), dtype=float)
        ym = np.asarray(f(xm), dtype=float)
        J[:, i] = (yp - ym) / (2.0 * eps)
    return J


class EKF:
    def __init__(
        self,
        f_state,              # x_{k+1} = f(x_k, u_k)
        h_meas,               # z_k = h(x_k)
        x0: np.ndarray,
        P0: Optional[np.ndarray] = None,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
    ):
        self.f_state = f_state
        self.h_meas = h_meas
        self.x = np.asarray(x0, dtype=float).copy()
        self.P = np.eye(self.x.size, dtype=float) if P0 is None else np.asarray(P0, dtype=float).copy()
        self.Q = np.eye(self.x.size, dtype=float) if Q is None else np.asarray(Q, dtype=float).copy()
        self.R = np.array([[1.0]], dtype=float) if R is None else np.asarray(R, dtype=float).copy()

    def predict(self, u: np.ndarray):
        u = np.asarray(u, dtype=float)

        def fx(xx):
            return self.f_state(xx, u)

        F = _num_jacobian(fx, self.x)
        self.x = np.asarray(self.f_state(self.x, u), dtype=float)
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z: float):
        z = float(z)

        def hx(xx):
            return np.array([self.h_meas(xx)], dtype=float)

        H = _num_jacobian(hx, self.x)  # shape (1,n)
        y = np.array([[z - self.h_meas(self.x)]], dtype=float)  # innovation (1,1)
        S = H @ self.P @ H.T + self.R  # (1,1)
        K = self.P @ H.T @ np.linalg.inv(S)  # (n,1)

        self.x = self.x + (K @ y).reshape(-1)

        # Joseph form for numerical stability
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.R @ K.T


# =============================================================================
# Cost with ControlHorizon=2 move-blocking
# U = [theta0, theta1, theta1, ..., theta1]
# =============================================================================

def plan_theta_pair_grid_fast(
    model: PondCSTR,
    xk: np.ndarray,
    q_forecast: np.ndarray,
    c_forecast: np.ndarray,
    theta_candidates: np.ndarray,
    hlimit_ft: float,
    warmstart: tuple[float, float],
) -> tuple[float, float]:

    cand = np.asarray(theta_candidates, dtype=float)
    n = cand.size

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

    sum_cq2 = np.zeros(P, dtype=float)   # sum((c*qout)^2)
    sum_q = np.zeros(P, dtype=float)     # sum(qout)
    sum_q2 = np.zeros(P, dtype=float)    # sum(qout^2)

    exp_kdt = np.exp(-model.k * model.dt)

    for j in range(H):
        theta = th0 if j == 0 else th1  # (P,)
        A = model.area_vec(h)           # (P,)
        qout = model.q_out_vec(h, theta)  # (P,)

        # cost accumulators use c BEFORE update
        cq = c * qout
        sum_cq2 += cq * cq
        sum_q += qout
        sum_q2 += qout * qout

        # state update
        q_in = qf[j]
        c_in = cf[j]

        h_next = h + (model.dt / A) * (q_in - qout)
        h_next = np.maximum(0.0, h_next)

        # concentration update uses current h (MATLAB: if h>0)
        den = (A * h) + q_in * model.dt
        den = np.maximum(den, 1e-6)
        c_next = np.where(
            h > 0.0,
            (c * A * h * exp_kdt + c_in * q_in * model.dt) / den,
            0.0,
        )

        # hard constraint
        violated |= (h_next > hlimit_ft)

        h, c = h_next, c_next

    qbar = sum_q / max(H, 1)
    smooth = sum_q2 - H * (qbar * qbar)
    cost = 5.0 * sum_cq2 + smooth + 900.0 * (h * h)
    cost[violated] = np.inf

    best = int(np.argmin(cost))
    return float(th0[best]), float(th1[best])


def plan_theta_pair_stochastic_fast(
    model,
    xk: np.ndarray,
    q_scen: np.ndarray,   # (H, Ns)
    c_scen: np.ndarray,   # (H, Ns)
    theta_candidates: np.ndarray,
    hlimit_ft: float,
    epsilon: float,
    warmstart: Tuple[float, float],
) -> Tuple[float, float]:
    """
    Fast scenario-based SMPC planning with move-blocking (CH=2).
    Evaluates all (theta0, theta1) pairs in a batch and all scenarios in parallel.

    Chance constraint: reject a control pair if fraction of scenarios that violate
    hlimit exceeds epsilon. A scenario is marked violated if h exceeds hlimit at any step.
    """
    cand = np.asarray(theta_candidates, dtype=float)
    n = cand.size

    # All (theta0, theta1) pairs: P = n*n
    th0_grid, th1_grid = np.meshgrid(cand, cand, indexing="ij")
    th0 = th0_grid.reshape(-1)   # (P,)
    th1 = th1_grid.reshape(-1)   # (P,)
    P = th0.size

    H, Ns = q_scen.shape
    q_scen = np.asarray(q_scen, dtype=float)
    c_scen = np.asarray(c_scen, dtype=float)

    # Batch states for all pairs and scenarios: shape (P, Ns)
    h = np.full((P, Ns), float(xk[0]), dtype=float)
    c = np.full((P, Ns), float(xk[1]), dtype=float)

    # Track per-scenario violation (any step violates => scenario violates)
    violated = np.zeros((P, Ns), dtype=bool)

    # Cost accumulators per (pair, scenario)
    sum_cq2 = np.zeros((P, Ns), dtype=float)  # sum((c*qout)^2)
    sum_q   = np.zeros((P, Ns), dtype=float)  # sum(qout)
    sum_q2  = np.zeros((P, Ns), dtype=float)  # sum(qout^2)

    exp_kdt = float(np.exp(-model.k * model.dt))

    # Pre-broadcast theta for j=0 and j>=1
    th0_b = th0[:, None]   # (P,1)
    th1_b = th1[:, None]   # (P,1)

    for j in range(H):
        theta = th0_b if j == 0 else th1_b  # (P,1) broadcast to (P,Ns)

        A = model.area_vec(h)               # (P,Ns)
        qout = model.q_out_vec(h, theta)    # (P,Ns)

        # Accumulate stage cost terms using c BEFORE update
        cq = c * qout
        sum_cq2 += cq * cq
        sum_q   += qout
        sum_q2  += qout * qout

        # Disturbances at step j: (Ns,)
        q_in = q_scen[j][None, :]  # (1,Ns)
        c_in = c_scen[j][None, :]  # (1,Ns)

        # State update
        h_next = h + (model.dt / A) * (q_in - qout)
        h_next = np.maximum(0.0, h_next)

        den = (A * h) + q_in * model.dt
        den = np.maximum(den, 1e-6)

        c_next = np.where(
            h > 0.0,
            (c * A * h * exp_kdt + c_in * q_in * model.dt) / den,
            0.0,
        )

        violated |= (h_next > hlimit_ft)

        h, c = h_next, c_next

    # Smoothness term: sum((q - mean(q))^2) = sum(q^2) - H*mean(q)^2
    qbar = sum_q / max(H, 1)
    smooth = sum_q2 - H * (qbar * qbar)

    # Full cost per (pair, scenario)
    J = 5.0 * sum_cq2 + smooth + 900.0 * (h * h)

    # Invalidate violated scenarios
    J[violated] = np.inf

    # Chance constraint: fraction of violated scenarios per pair
    prob_violate = violated.mean(axis=1)  # (P,)
    feasible = prob_violate <= float(epsilon)

    # Expected cost over NON-violated scenarios (feasible pairs only)
    # (If all scenarios violated -> inf)
    J_exp = np.full(P, np.inf, dtype=float)
    if np.any(feasible):
        J_f = J[feasible]  # (P_f, Ns)
        finite = np.isfinite(J_f)
        cnt = finite.sum(axis=1)
        # Avoid division by zero
        mean_cost = np.where(cnt > 0, np.sum(np.where(finite, J_f, 0.0), axis=1) / cnt, np.inf)
        J_exp[feasible] = mean_cost

    best = int(np.argmin(J_exp))
    return float(th0[best]), float(th1[best])


# =============================================================================
# Simulators / controllers
# =============================================================================

def simulate_passive_system(model: PondCSTR, x0: np.ndarray, MD_t: np.ndarray) -> PassiveResult:
    Tsim = MD_t.shape[0]
    Xopt = np.zeros((Tsim, 2), dtype=float)
    Yopt = np.zeros((Tsim, 3), dtype=float)
    Xopt[0] = x0

    u0 = np.array([1.0, MD_t[0, 0], MD_t[0, 1]], dtype=float)
    Yopt[0] = model.output(x0, u0)

    for k in range(Tsim - 1):
        u = np.array([1.0, MD_t[k, 0], MD_t[k, 1]], dtype=float)
        Xopt[k + 1] = model.state_step(Xopt[k], u)
        Yopt[k + 1] = model.output(Xopt[k], u)

    return PassiveResult(Xopt=Xopt, Yopt=Yopt)


def run_deterministic_mpc(
    model: PondCSTR,
    x0: np.ndarray,
    u0: float,
    qin_forecast: np.ndarray,
    cin_forecast: np.ndarray,
    MD_exec_true: np.ndarray,
    horizon: int,
    theta_candidates: np.ndarray,
    hlimit_ft: float,
) -> ControllerTraj:
    """
    Deterministic MPC with CH=2 move-blocking, grid search over candidates.
    Planning uses (qin_forecast, cin_forecast).
    Execution uses true measured disturbances MD_exec_true = [qin_t, cin_t].
    """
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

        th0, th1 = plan_theta_pair_grid_fast(
            model=model,
            xk=np.array([h[k], c[k]], dtype=float),
            q_forecast=q_base,
            c_forecast=c_base,
            theta_candidates=theta_candidates,
            hlimit_ft=hlimit_ft,
            warmstart=warm,
        )

        # apply first move (like nlmpcmove returning MVopt(2,:) loop)
        theta_apply = th0
        theta[k] = theta_apply

        u_real = np.array([theta_apply, MD_exec_true[k, 0], MD_exec_true[k, 1]], dtype=float)
        yk = model.output(np.array([h[k], c[k]], dtype=float), u_real)
        qout[k] = yk[2]

        x_next = model.state_step(np.array([h[k], c[k]], dtype=float), u_real)
        h[k + 1], c[k + 1] = float(x_next[0]), float(x_next[1])

        # warmstart next step with second move
        warm = (th1, th1)
        theta[k + 1] = th1  # optional warmstart storage

    # final qout fill
    for k in range(Tsim - 1, -1, -1):
        if qout[k] == 0.0 and k > 0:
            u_real = np.array([theta[k], MD_exec_true[k, 0], MD_exec_true[k, 1]], dtype=float)
            qout[k] = model.output(np.array([h[k], c[k]], dtype=float), u_real)[2]

    return ControllerTraj(h_ft=h, c=c, qout_cfs=qout, theta=theta, qspill_cfs=None)


def run_mpc_ekf(
    model: PondCSTR,
    x0: np.ndarray,
    u0: float,
    qin_forecast: np.ndarray,
    cin_forecast: np.ndarray,
    MD_exec_true: np.ndarray,
    horizon: int,
    theta_candidates: np.ndarray,
    hlimit_ft: float,
    meas_noise_std: float = 0.0,   # set >0 for noisy measurement
    rng: Optional[np.random.Generator] = None,
) -> ControllerTraj:
    """
    MPC-EKF: use EKF to estimate state, then MPC plans using the estimate + imperfect forecasts.
    Measurement: z = c_true + noise
    """
    if rng is None:
        rng = np.random.default_rng(0)

    Tsim = len(qin_forecast)

    # true plant state
    h_true = np.zeros(Tsim, dtype=float)
    c_true = np.zeros(Tsim, dtype=float)
    q_true = np.zeros(Tsim, dtype=float)
    theta = np.zeros(Tsim, dtype=float)

    h_true[0], c_true[0] = float(x0[0]), float(x0[1])
    theta[0] = float(np.clip(u0, 0.0, 1.0))

    # EKF setup 
    def f_state(xx: np.ndarray, uu: np.ndarray) -> np.ndarray:
        return model.state_step(xx, uu)

    def h_meas(xx: np.ndarray) -> float:
        return float(xx[1])  # measure concentration

    Q = np.diag([0.0, 1.0]) 
    R = np.array([[0.1]], dtype=float)  # like MeasurementNoise = 0.1
    ekf = EKF(f_state=f_state, h_meas=h_meas, x0=x0, P0=None, Q=Q, R=R)

    warm = (theta[0], theta[0])

    for k in range(Tsim - horizon - 1):
        # (1) measurement from true plant (after applying previous control)
        z = c_true[k] + (meas_noise_std * rng.standard_normal())

        # (2) EKF update with measurement
        ekf.update(z)

        # (3) MPC planning with current estimate and forecast MD (possibly imperfect)
        q_base = qin_forecast[k : k + horizon]
        c_base = cin_forecast[k : k + horizon]
        th0, th1 = plan_theta_pair_grid_fast(
            model=model,
            xk=ekf.x.copy(),
            q_forecast=q_base,
            c_forecast=c_base,
            theta_candidates=theta_candidates,
            hlimit_ft=hlimit_ft,
            warmstart=warm,
        )

        theta_apply = th0
        theta[k] = theta_apply

        # (4) apply control to true plant using true disturbances
        u_true = np.array([theta_apply, MD_exec_true[k, 0], MD_exec_true[k, 1]], dtype=float)
        q_true[k] = model.output(np.array([h_true[k], c_true[k]], dtype=float), u_true)[2]
        x_next_true = model.state_step(np.array([h_true[k], c_true[k]], dtype=float), u_true)
        h_true[k + 1], c_true[k + 1] = float(x_next_true[0]), float(x_next_true[1])

        # (5) EKF predict with the same applied input and measured (true) disturbances
        ekf.predict(u_true)

        warm = (th1, th1)
        theta[k + 1] = th1

    # fill remaining qout
    for k in range(Tsim - 1, -1, -1):
        if q_true[k] == 0.0 and k > 0:
            u_true = np.array([theta[k], MD_exec_true[k, 0], MD_exec_true[k, 1]], dtype=float)
            q_true[k] = model.output(np.array([h_true[k], c_true[k]], dtype=float), u_true)[2]

    return ControllerTraj(h_ft=h_true, c=c_true, qout_cfs=q_true, theta=theta, qspill_cfs=None)


def run_stochastic_mpc_fast(
    model,
    x0: np.ndarray,
    Ns: int,
    horizon: int,
    theta_candidates: np.ndarray,
    epsilon: float,
    sigma_q: float,
    sigma_c: float,
    MD_exec_true: np.ndarray,
    qin_forecast: np.ndarray,
    cin_forecast: np.ndarray,
    hlimit_ft: float,
    rng: np.random.Generator,
) -> ControllerTraj:
    """
    Fast scenario-based SMPC (vectorized over scenarios and theta-pairs).
    """
    Tsim = len(qin_forecast)

    h = np.zeros(Tsim, dtype=float)
    c = np.zeros(Tsim, dtype=float)
    qout = np.zeros(Tsim, dtype=float)
    theta = np.zeros(Tsim, dtype=float)

    h[0], c[0] = float(x0[0]), float(x0[1])
    theta[0] = 1.0
    warm = (theta[0], theta[0])

    for k in range(Tsim - horizon - 1):
        q_base = qin_forecast[k : k + horizon].astype(float)
        c_base = cin_forecast[k : k + horizon].astype(float)

        # Scenario generation: multiplicative noise, clamp to >=0
        base_q = q_base[:, None]  # (H,1)
        base_c = c_base[:, None]  # (H,1)

        q_scen = np.maximum(
            0.0, base_q * (1.0 + sigma_q * rng.standard_normal((horizon, Ns)))
        )
        c_scen = np.maximum(
            0.0, base_c * (1.0 + sigma_c * rng.standard_normal((horizon, Ns)))
        )

        th0, th1 = plan_theta_pair_stochastic_fast(
            model=model,
            xk=np.array([h[k], c[k]], dtype=float),
            q_scen=q_scen,
            c_scen=c_scen,
            theta_candidates=theta_candidates,
            hlimit_ft=hlimit_ft,
            epsilon=epsilon,
            warmstart=warm,
        )

        # Apply the first move to the real plant (true disturbances)
        theta_apply = th0
        theta[k] = theta_apply

        u_real = np.array([theta_apply, MD_exec_true[k, 0], MD_exec_true[k, 1]], dtype=float)
        qout[k] = model.output(np.array([h[k], c[k]], dtype=float), u_real)[2]

        x_next = model.state_step(np.array([h[k], c[k]], dtype=float), u_real)
        h[k + 1], c[k + 1] = float(x_next[0]), float(x_next[1])

        # Warm-start: shift to the second move
        warm = (th1, th1)
        theta[k + 1] = th1

    # Fill remaining qout for trailing indices
    for k in range(Tsim - 1, -1, -1):
        if qout[k] == 0.0 and k > 0:
            u_real = np.array([theta[k], MD_exec_true[k, 0], MD_exec_true[k, 1]], dtype=float)
            qout[k] = model.output(np.array([h[k], c[k]], dtype=float), u_real)[2]

    return ControllerTraj(h_ft=h, c=c, qout_cfs=qout, theta=theta, qspill_cfs=None)


# =============================================================================
# RBC baselines
# =============================================================================

def compute_overflow_flow_cfs(h_ft: np.ndarray, hmax_ft: float, Amax_ft2: float, dt_s: float) -> np.ndarray:
    """
    Approximate overflow as:
      q_spill = Amax * max(h - hmax, 0) / dt
    """
    depth = np.maximum(0.0, h_ft - hmax_ft)
    vol_ft3 = Amax_ft2 * depth
    qspill = vol_ft3 / dt_s
    return qspill


def rbc_outflow(
    model: PondCSTR,
    MD_t: np.ndarray,
    x0: np.ndarray,
    q_desired_cfs: float,
    hlimit_ft: float,
    h_reten: float = 0.05,
) -> ControllerTraj:
    co, Ao, g = model.co, model.Ao, model.g

    T = MD_t.shape[0]
    h = np.zeros(T + 1, dtype=float)
    c = np.zeros(T + 1, dtype=float)
    qout = np.zeros(T, dtype=float)
    theta = np.zeros(T, dtype=float)

    h[0], c[0] = float(x0[0]), float(x0[1])
    indicator = 1

    hdes = (1.0 / (2.0 * g)) * (q_desired_cfs / (co * Ao)) ** 2 

    for t in range(T):
        q_in, c_in = float(MD_t[t, 0]), float(MD_t[t, 1])

        # update concentration using current state
        A = model.area(h[t])
        if h[t] > 0:
            c[t + 1] = (c[t] * A * h[t] * np.exp(-model.k * model.dt) + c_in * q_in * model.dt) / (A * h[t] + q_in * model.dt)
        else:
            c[t + 1] = 0.0

        if (h[t] < hlimit_ft) and (indicator == 1):
            qout[t] = 0.0
            theta[t] = 0.0
            h[t + 1] = h[t] + model.dt / A * (q_in - qout[t])
            indicator = 1

        elif h[t] >= hlimit_ft:
            qout[t] = q_desired_cfs
            theta[t] = 1.0
            h[t + 1] = max(0.0, h[t] + model.dt / A * (q_in - qout[t]))
            indicator = 0

        elif (h[t] < hlimit_ft) and (h[t] >= hdes) and (indicator == 0):
            qout[t] = q_desired_cfs
            theta[t] = q_desired_cfs / (co * Ao * np.sqrt(2.0 * g * max(h[t], 1e-9)))
            h[t + 1] = max(0.0, h[t] + model.dt / A * (q_in - qout[t]))
            indicator = 0

        elif (h[t] < hdes) and (indicator == 0) and (h[t] > h_reten):
            qout[t] = co * Ao * np.sqrt(2.0 * g * h[t]) * min(1.0, h[t])
            theta[t] = 1.0
            h[t + 1] = max(0.0, h[t] + model.dt / A * (q_in - qout[t]))
            indicator = 0

        else:  # h <= h_reten and indicator==0
            qout[t] = 0.0
            theta[t] = 0.0
            h[t + 1] = h[t] + model.dt / A * (q_in - qout[t])
            indicator = 1

    # align lengths back to T
    h = h[:T]
    c = c[:T]

    # overflow spill flow (cfs)
    Amax_ft2 = float(np.max(model.area_tab))
    qspill = compute_overflow_flow_cfs(h, hmax_ft=10.0, Amax_ft2=Amax_ft2, dt_s=model.dt)

    return ControllerTraj(h_ft=h, c=c, qout_cfs=qout, theta=theta, qspill_cfs=qspill)


def rbc_concentration(
    model: PondCSTR,
    MD_t: np.ndarray,
    x0: np.ndarray,
    climit: float,
    hlimit_ft: float,
) -> ControllerTraj:
    co, Ao, g = model.co, model.Ao, model.g

    T = MD_t.shape[0]
    h = np.zeros(T + 1, dtype=float)
    c = np.zeros(T + 1, dtype=float)
    qout = np.zeros(T, dtype=float)
    theta = np.zeros(T, dtype=float)

    h[0], c[0] = float(x0[0]), float(x0[1])

    for t in range(T):
        q_in, c_in = float(MD_t[t, 0]), float(MD_t[t, 1])
        A = model.area(h[t])

        if h[t] > 0:
            c[t + 1] = (c[t] * A * h[t] * np.exp(-model.k * model.dt) + c_in * q_in * model.dt) / (A * h[t] + q_in * model.dt)
        else:
            c[t + 1] = 0.0

        if (c[t] > climit) and (h[t] < hlimit_ft):
            qout[t] = 0.0
            theta[t] = 0.0
            h[t + 1] = h[t] + model.dt / A * (q_in - qout[t])

        elif h[t] >= hlimit_ft:
            qout[t] = co * Ao * np.sqrt(2.0 * g * h[t]) * min(1.0, h[t])
            theta[t] = 1.0
            h[t + 1] = max(0.0, h[t] + model.dt / A * (q_in - qout[t]))

        else:  # c[t] < climit
            qout[t] = co * Ao * np.sqrt(2.0 * g * h[t]) * min(1.0, h[t])
            theta[t] = 1.0
            h[t + 1] = max(0.0, h[t] + model.dt / A * (q_in - qout[t]))

    h = h[:T]
    c = c[:T]
    Amax_ft2 = float(np.max(model.area_tab))
    qspill = compute_overflow_flow_cfs(h, hmax_ft=10.0, Amax_ft2=Amax_ft2, dt_s=model.dt)
    return ControllerTraj(h_ft=h, c=c, qout_cfs=qout, theta=theta, qspill_cfs=qspill)


def rbc_both(
    model: PondCSTR,
    MD_t: np.ndarray,
    x0: np.ndarray,
    climit: float,
    q_desired_cfs: float,
    hlimit_ft: float,
) -> ControllerTraj:
    co, Ao, g = model.co, model.Ao, model.g

    T = MD_t.shape[0]
    h = np.zeros(T + 1, dtype=float)
    c = np.zeros(T + 1, dtype=float)
    qout = np.zeros(T, dtype=float)
    theta = np.zeros(T, dtype=float)

    h[0], c[0] = float(x0[0]), float(x0[1])

    hdes = (1.0 / (2.0 * g)) * (q_desired_cfs / (co * Ao)) ** 2

    for t in range(T):
        q_in, c_in = float(MD_t[t, 0]), float(MD_t[t, 1])
        A = model.area(h[t])

        if h[t] > 0:
            c[t + 1] = (c[t] * A * h[t] * np.exp(-model.k * model.dt) + c_in * q_in * model.dt) / (A * h[t] + q_in * model.dt)
        else:
            c[t + 1] = 0.0

        if (c[t] > climit) and (h[t] < hlimit_ft):
            qout[t] = 0.0
            theta[t] = 0.0
            h[t + 1] = h[t] + model.dt / A * (q_in - qout[t])

        elif h[t] >= hlimit_ft:
            qout[t] = q_desired_cfs
            theta[t] = 1.0
            h[t + 1] = max(0.0, h[t] + model.dt / A * (q_in - qout[t]))

        else:  # c[t] < climit
            if (h[t] < hlimit_ft) and (h[t] >= hdes):
                qout[t] = q_desired_cfs
                theta[t] = q_desired_cfs / (co * Ao * np.sqrt(2.0 * g * max(h[t], 1e-9)))
                h[t + 1] = max(0.0, h[t] + model.dt / A * (q_in - qout[t]))
            else:  # h < hdes
                qout[t] = co * Ao * np.sqrt(2.0 * g * h[t]) * min(1.0, h[t])
                theta[t] = 1.0
                h[t + 1] = max(0.0, h[t] + model.dt / A * (q_in - qout[t]))

    h = h[:T]
    c = c[:T]
    Amax_ft2 = float(np.max(model.area_tab))
    qspill = compute_overflow_flow_cfs(h, hmax_ft=10.0, Amax_ft2=Amax_ft2, dt_s=model.dt)
    return ControllerTraj(h_ft=h, c=c, qout_cfs=qout, theta=theta, qspill_cfs=qspill)


# =============================================================================
# Metrics & reporting
# =============================================================================

def compute_metrics(
    trajs: Dict[str, ControllerTraj],
    horizon: int,
    model: PondCSTR,
) -> pd.DataFrame:
    # US->SI
    ft_to_m = 1.0 / 3.281
    ft2_to_m2 = 1.0 / 10.764
    cfs_to_m3s = 0.028316846592

    # pond geometry
    Amax_ft2 = float(np.max(model.area_tab))
    Amax_m2 = Amax_ft2 * ft2_to_m2
    h_over_ft = 10.0
    h_over_m = h_over_ft * ft_to_m

    # evaluate window consistent: length(MD) - horizon
    # (avoid the appended post-storm zeros influencing end)
    any_traj = next(iter(trajs.values()))
    T = len(any_traj.h_ft)
    T_eval = max(1, T - horizon)
    idx = slice(0, T_eval)

    rows: List[Dict[str, float]] = []

    # precompute maxima for percent normalization
    overflow_list = []
    peak_list = []
    load_list = []
    effort_list = []
    smooth_list = []

    raw = {}

    for name, tr in trajs.items():
        h_m = tr.h_ft[idx] * ft_to_m
        q_m3s = tr.qout_cfs[idx] * cfs_to_m3s
        c = tr.c[idx]
        th = tr.theta[idx]

        # overflow volume (m^3)
        overflow_vol = max(0.0, (float(np.max(h_m)) - h_over_m) * Amax_m2)

        # peak outflow
        peak_q = float(np.max(q_m3s))

        # cumulative load (sum(C*q)*1e-3*dt, optionally add spill)
        # If RBC provides qspill, include spill load term: C*qspill
        if tr.qspill_cfs is not None:
            qspill_m3s = tr.qspill_cfs[idx] * cfs_to_m3s
            load = float(np.sum(c * (q_m3s + qspill_m3s)) * 1e-3 * model.dt)
        else:
            load = float(np.sum(c * q_m3s) * 1e-3 * model.dt)

        # control effort
        effort = float(np.sum(np.diff(th) ** 2)) if len(th) > 1 else 0.0

        # outflow smoothness
        smooth = float(np.sum((q_m3s - float(np.mean(q_m3s))) ** 2))

        raw[name] = dict(
            overflow_m3=overflow_vol,
            peak_m3s=peak_q,
            load=load,
            effort=effort,
            smooth=smooth,
        )

        overflow_list.append(overflow_vol)
        peak_list.append(peak_q)
        load_list.append(load)
        effort_list.append(effort)
        smooth_list.append(smooth)

    def pct(val: float, vec: List[float]) -> float:
        m = max(vec) if max(vec) > 0 else 1.0
        return 100.0 * val / m

    for name in trajs.keys():
        r = raw[name]
        rows.append(
            dict(
                Controller=name,
                Overflow_m3=r["overflow_m3"],
                Overflow_pct=pct(r["overflow_m3"], overflow_list),
                PeakQ_m3s=r["peak_m3s"],
                PeakQ_pct=pct(r["peak_m3s"], peak_list),
                Load=r["load"],
                Load_pct=pct(r["load"], load_list),
                Effort=r["effort"],
                Effort_pct=pct(r["effort"], effort_list),
                Smooth=r["smooth"],
                Smooth_pct=pct(r["smooth"], smooth_list),
            )
        )

    df = pd.DataFrame(rows).set_index("Controller")
    return df


def print_metrics_table(df: pd.DataFrame) -> None:
    cols = ["Overflow_m3", "PeakQ_m3s", "Load", "Effort", "Smooth"]
    print("\n=== Performance comparison (SI units) ===")
    print(df[cols].to_string(float_format=lambda x: f"{x:,.4g}"))

    cols_pct = ["Overflow_pct", "PeakQ_pct", "Load_pct", "Effort_pct", "Smooth_pct"]
    print("\n=== Normalized (percent of worst) ===")
    print(df[cols_pct].to_string(float_format=lambda x: f"{x:6.2f}"))


def save_results(
    out_path: Path,
    trajs: Dict[str, ControllerTraj],
    metrics: pd.DataFrame,
    qin_t: np.ndarray,
    cin_t: np.ndarray,
    qin_f: np.ndarray,
    cin_f: np.ndarray,
) -> None:
    data = {
        "qin_t": qin_t[:, None],
        "cin_t": cin_t[:, None],
        "qin_f": qin_f[:, None],
        "cin_f": cin_f[:, None],
        "metrics": metrics.reset_index().to_dict(orient="list"),
    }

    for name, tr in trajs.items():
        key = name.replace(" ", "_").replace("-", "_")
        data[f"{key}_h"] = tr.h_ft[:, None]
        data[f"{key}_c"] = tr.c[:, None]
        data[f"{key}_qout"] = tr.qout_cfs[:, None]
        data[f"{key}_theta"] = tr.theta[:, None]
        if tr.qspill_cfs is not None:
            data[f"{key}_qspill"] = tr.qspill_cfs[:, None]

    if savemat is not None:
        savemat(str(out_path), data)
        print(f"\nSaved results to: {out_path}")
    else:
        np.savez(out_path.with_suffix(".npz"), **data)
        print(f"\nSciPy not available. Saved results to: {out_path.with_suffix('.npz')}")


# =============================================================================
# Main benchmark
# =============================================================================

def mpc_benchmark() -> None:
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "pond4_background.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data file: {data_path}")

    data1 = pd.read_csv(data_path)  # true
    data2 = pd.read_csv(data_path)  # forecasts

    extra_len = 789
    zeros = np.zeros(extra_len, dtype=float)

    qin_t = np.concatenate([data1["qin"].to_numpy(dtype=float), zeros])
    cin_t = np.concatenate([data1["cin"].to_numpy(dtype=float), zeros])

    qin_f = np.concatenate([data2["qin"].to_numpy(dtype=float), zeros])

    # Imperfect water quality prediction as EMC
    EMC = float(np.sum(data1["qin"].to_numpy(dtype=float) * data1["cin"].to_numpy(dtype=float)) / max(np.sum(data1["qin"].to_numpy(dtype=float)), 1e-9))
    cin_f = np.full_like(qin_t, EMC, dtype=float)

    MD_t = np.column_stack((qin_t, cin_t))  # measured disturbances (true)
    MD_f = np.column_stack((qin_f, cin_f))  # forecast disturbances (imperfect)
    # MD (forecast) is (qin_f, cin_f) implicitly via arrays

    # model (spline interpolation if SciPy installed)
    model = PondCSTR(use_spline=True)

    horizon = 96  # 24 hr at 15-min steps
    theta_candidates = np.linspace(0.0, 1.0, 21)

    x0 = np.array([0.01, 0.0], dtype=float)
    u0 = 1.0
    hlimit_ft = 10.0

    # stochastic settings
    # Ns = 10
    # epsilon = 0.05
    # sigma_q = 0.30
    # sigma_c = 0.30
    sigma_q = 0.05
    sigma_c = 0.05
    epsilon = 0.05
    Ns = 10

    # Use these will make stochastic MPC equivalent to deterministic MPC
    # sigma_q = 0.0
    # sigma_c = 0.0
    # epsilon = 1.0
    # Ns = 1

    rng = np.random.default_rng(seed=0)

    # ------------------------------------------------------------------
    # Passive
    # ------------------------------------------------------------------
    print("\nPassive system simulation (theta=1)...")
    passive = simulate_passive_system(model, x0, MD_t)
    traj_passive = ControllerTraj(
        h_ft=passive.Xopt[:, 0],
        c=passive.Xopt[:, 1],
        qout_cfs=passive.Yopt[:, 2],
        theta=np.ones_like(passive.Xopt[:, 0]),
        qspill_cfs=compute_overflow_flow_cfs(passive.Xopt[:, 0], 10.0, float(np.max(model.area_tab)), model.dt),
    )

    # # # ------------------------------------------------------------------
    # # # MPC-false (imperfect forecast concentration cin_f=EMC)
    # # # ------------------------------------------------------------------
    print("\nDeterministic MPC-false simulation...")
    t0 = time.perf_counter()
    traj_mpc_false = run_deterministic_mpc(
        model=model,
        x0=x0,
        u0=u0,
        qin_forecast=qin_f,
        cin_forecast=cin_f,
        MD_exec_true=MD_f,
        horizon=horizon,
        theta_candidates=theta_candidates,
        hlimit_ft=hlimit_ft,
    )
    print(f"MPC-false done in {time.perf_counter() - t0:.2f}s")

    # ------------------------------------------------------------------
    # MPC-true (perfect forecast uses MD_t)
    # ------------------------------------------------------------------
    print("\nDeterministic MPC-true simulation...")
    t0 = time.perf_counter()
    traj_mpc_true = run_deterministic_mpc(
        model=model,
        x0=x0,
        u0=u0,
        qin_forecast=qin_t,
        cin_forecast=cin_t,
        MD_exec_true=MD_t,
        horizon=horizon,
        theta_candidates=theta_candidates,
        hlimit_ft=hlimit_ft,
    )
    print(f"MPC-true done in {time.perf_counter() - t0:.2f}s")

    # # ------------------------------------------------------------------
    # # MPC-EKF (plan with imperfect forecast; EKF uses c measurement)
    # # ------------------------------------------------------------------
    print("\nMPC-EKF simulation (no measurement noise by default)...")
    t0 = time.perf_counter()
    traj_mpc_ekf = run_mpc_ekf(
        model=model,
        x0=x0,
        u0=u0,
        qin_forecast=qin_f,
        cin_forecast=cin_f,
        MD_exec_true=MD_t,
        horizon=horizon,
        theta_candidates=theta_candidates,
        hlimit_ft=hlimit_ft,
        meas_noise_std=0.0,
        rng=rng,
    )
    print(f"MPC-EKF done in {time.perf_counter() - t0:.2f}s")

    # ------------------------------------------------------------------
    # Stochastic MPC (scenario-based, chance constraint)
    # ------------------------------------------------------------------
    print("\nStochastic MPC simulation...")
    t0 = time.perf_counter()
    traj_smpc = run_stochastic_mpc_fast(
        model=model,
        x0=x0,
        Ns=Ns,
        horizon=horizon,
        theta_candidates=theta_candidates,
        epsilon=epsilon,
        sigma_q=sigma_q,
        sigma_c=sigma_c,
        MD_exec_true=MD_t,
        qin_forecast=qin_f,
        cin_forecast=cin_f,
        hlimit_ft=hlimit_ft,
        rng=rng,
    )
    print(f"Stochastic MPC done in {time.perf_counter() - t0:.2f}s")

    # ------------------------------------------------------------------
    # RBC baselines (use q_desired = max(truey))
    # ------------------------------------------------------------------
    print("\nRBC simulations...")
    q_desired_cfs = float(np.max(traj_mpc_true.qout_cfs))
    climit = 5.0

    traj_rbc_out = rbc_outflow(model, MD_t, x0, q_desired_cfs=q_desired_cfs, hlimit_ft=float(np.max(traj_mpc_true.h_ft)))
    traj_rbc_con = rbc_concentration(model, MD_t, x0, climit=climit, hlimit_ft=float(np.max(traj_mpc_true.h_ft)))
    traj_rbc_both = rbc_both(model, MD_t, x0, climit=climit, q_desired_cfs=q_desired_cfs, hlimit_ft=float(np.max(traj_mpc_true.h_ft)))

    # ------------------------------------------------------------------
    # Compare
    # ------------------------------------------------------------------
    trajs = {
        "MPC-true": traj_mpc_true,
        "MPC-false": traj_mpc_false,
        "MPC-EKF": traj_mpc_ekf,
        "Stochastic MPC": traj_smpc,
        "RBC-Concentration": traj_rbc_con,
        "RBC-Outflow": traj_rbc_out,
        "RBC-Both": traj_rbc_both,
        "Passive": traj_passive,
    }

    metrics = compute_metrics(trajs, horizon=horizon, model=model)
    print_metrics_table(metrics)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_path = base_dir / "mpc_benchmark_results.mat"
    save_results(out_path, trajs, metrics, qin_t, cin_t, qin_f, cin_f)


if __name__ == "__main__":
    mpc_benchmark()
