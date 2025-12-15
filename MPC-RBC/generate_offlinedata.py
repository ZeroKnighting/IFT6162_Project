import numpy as np
from typing import Tuple, Dict, Any, Optional
from pathlib import Path
import pandas as pd
import time
from benchmark import (
    plan_theta_pair_grid_fast_with_cost, 
    PondCSTR, 
    simulate_passive_system,
    run_deterministic_mpc,
    run_stochastic_mpc_fast,
    ControllerTraj,
    compute_overflow_flow_cfs,
    compute_metrics,
    print_metrics_table,
    save_results,
)


def collect_lao_dataset_rollouts(
    model,
    qin_forecast: np.ndarray,
    cin_forecast: np.ndarray,
    MD_exec_true: np.ndarray,
    horizon: int,
    theta_candidates: np.ndarray,
    hlimit_ft: float,
    n_episodes: int = 20,
    episode_len: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    h0_range: Tuple[float, float] = (0.0, 2.0),
    c0_range: Tuple[float, float] = (0.0, 10.0),
) -> Dict[str, Any]:
    """
    Collect LAO training dataset via rollouts.
    input x: current state [h,c]
    output: optimal control pair [theta0*, theta1*], optimal cost J*

    rollout use true disturbance (MD_exec_true),
    planning use forecast (qin_forecast, cin_forecast) (same pattern as your MPC-false).

    if episode_len is given, use that length;
    if episode_len is None, run until maximum possible length (T - horizon - 1).

  
    """
    if rng is None:
        rng = np.random.default_rng(0)

    T = len(qin_forecast)
    max_k = T - horizon - 1
    if max_k <= 1:
        raise ValueError("horizon is too long for the forecast length")

    if episode_len is None:
        episode_len = max_k
    episode_len = int(min(episode_len, max_k))

    X_list = []
    Ypi_list = []
    YV_list = []

    for ep in range(n_episodes):
        # random initial state
        h0 = rng.uniform(h0_range[0], h0_range[1])
        c0 = rng.uniform(c0_range[0], c0_range[1])
        x = np.array([h0, c0], dtype=float)

        warm = (1.0, 1.0)

        for k in range(episode_len):
            q_base = qin_forecast[k : k + horizon]
            c_base = cin_forecast[k : k + horizon]

            th0, th1, Jstar = plan_theta_pair_grid_fast_with_cost(
                model=model,
                xk=x,
                q_forecast=q_base,
                c_forecast=c_base,
                theta_candidates=theta_candidates,
                hlimit_ft=hlimit_ft,
                warmstart=warm,
                terminal_value_fn=None,  
                terminal_weight=1.0,
            )

            # store data
            X_list.append(x.copy())
            Ypi_list.append([th0, th1])
            YV_list.append([Jstar])

            # TRUE disturbance
            u_real = np.array([th0, MD_exec_true[k, 0], MD_exec_true[k, 1]], dtype=float)
            x = model.state_step(x, u_real)

            warm = (th1, th1)

    X = np.asarray(X_list, dtype=float)
    Y_pi = np.asarray(Ypi_list, dtype=float)
    Y_V = np.asarray(YV_list, dtype=float)

    return {
        "X": X,           # (N,2)
        "Y_pi": Y_pi,     # (N,2)
        "Y_V": Y_V,       # (N,1)
        "meta": {
            "n_episodes": n_episodes,
            "episode_len": episode_len,
            "horizon": horizon,
            "theta_candidates": np.asarray(theta_candidates, float),
            "hlimit_ft": float(hlimit_ft),
            "h0_range": h0_range,
            "c0_range": c0_range,
        }
    }


def save_offline_data(
    out_path: str = "pond4_lao_offline_data.npz",
    val_ratio: float = 0.2,
    normalize_X: bool = True,
    seed: int = 0,
) -> None:
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "pond4_background.csv"
    out_path = base_dir / out_path
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
    # MD (forecast) is (qin_f, cin_f) implicitly via arrays

    # model (spline interpolation if SciPy installed)
    model = PondCSTR(use_spline=True)

    horizon = 96  # 24 hr at 15-min steps
    theta_candidates = np.linspace(0.0, 1.0, 11)

    x0 = np.array([0.01, 0.0], dtype=float)
    u0 = 1.0
    hlimit_ft = 10.0

    # stochastic settings
    Ns = 10
    epsilon = 0.05
    sigma_q = 0.30
    sigma_c = 0.30
    # sigma_q = 0.01
    # sigma_c = 0.01
    # epsilon = 0.05
    # Ns = 10

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

    print("\nCollecting LAO training data...")
    dataset = collect_lao_dataset_rollouts(
        model=model,
        qin_forecast=qin_f,
        cin_forecast=cin_f,
        MD_exec_true=MD_t,
        horizon=horizon,
        theta_candidates=theta_candidates,
        hlimit_ft=hlimit_ft,
        n_episodes=10,
        episode_len=1500,
        rng=rng,
    )

    
    X = np.asarray(dataset["X"], dtype=float)
    print(f"Collected {X.shape[0]} samples for LAO")
    Ypi = np.asarray(dataset["Y_pi"], dtype=float)
    YV = np.asarray(dataset["Y_V"], dtype=float)


    N = X.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)

    n_val = int(np.floor(val_ratio * N))
    idx_val = perm[:n_val]
    idx_train = perm[n_val:]

    X_train, Ypi_train, YV_train = X[idx_train], Ypi[idx_train], YV[idx_train]
    X_val,   Ypi_val,   YV_val   = X[idx_val],   Ypi[idx_val],   YV[idx_val]

    if normalize_X:
        X_mean = X_train.mean(axis=0, keepdims=True)
        X_std  = X_train.std(axis=0, keepdims=True) + 1e-8

        X_train_n = (X_train - X_mean) / X_std
        X_val_n   = (X_val   - X_mean) / X_std
    else:
        X_mean = np.zeros((1, X.shape[1]), dtype=float)
        X_std  = np.ones((1, X.shape[1]), dtype=float)
        X_train_n, X_val_n = X_train, X_val

    out_path = Path(out_path)
    meta = dataset.get("meta", {})
    meta_repr = repr(meta).encode("utf-8")

    np.savez(
        out_path,
        X_train=X_train_n, Ypi_train=Ypi_train, YV_train=YV_train,
        X_val=X_val_n,     Ypi_val=Ypi_val,     YV_val=YV_val,
        X_mean=X_mean, X_std=X_std,
        idx_train=idx_train, idx_val=idx_val,
        meta_repr=meta_repr,
    )




if __name__ == "__main__":
    save_offline_data()