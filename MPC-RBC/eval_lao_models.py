"""
eval_lao_models.py

Evaluate trained LAO PolicyNet/ValueNet checkpoint against a saved LAO dataset (.npz).

What it does
- Loads dataset (expects keys from split_and_save_npz: X_val, Ypi_val, YV_val, X_mean, X_std)
- Loads model checkpoint produced by train_lao_nets.py (policy/value state_dict + v_mean/v_std + value_transform)
- Runs inference on validation set
- Reports:
    * Policy: MSE/MAE on (theta0, theta1), plus per-dimension metrics
    * Value: MSE/MAE in both transformed space (z-scored log1p) and original cost space (J*)
    * Value correlation (Pearson) in original cost space
- Optionally writes scatter plots to disk

Usage:
  python eval_lao_models.py --data lao_dataset.npz --ckpt lao_models.pt
  python eval_lao_models.py --data lao_dataset.npz --ckpt lao_models.pt --plots

Notes:
- This script assumes X_val in the npz is ALREADY normalized (as in your training script).
  It still reads X_mean/X_std for completeness, but does not renormalize.
- Non-finite YV samples are filtered out, consistent with training.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception as e:
    raise ImportError("This script requires PyTorch. Install with: pip install torch") from e



# ---------------------------
# IO
# ---------------------------

def load_npz(path: str | Path) -> Dict[str, np.ndarray]:
    data = np.load(Path(path), allow_pickle=False)
    return {k: data[k] for k in data.files}


def load_ckpt(path: str | Path, map_location: str = "cpu") -> Dict:
    return torch.load(Path(path), map_location=map_location)


# ---------------------------
# Value transforms
# ---------------------------

def value_forward_transform(J: np.ndarray, v_mean: np.ndarray, v_std: np.ndarray) -> np.ndarray:
    """raw J -> standardized log1p space"""
    J = np.asarray(J, dtype=np.float64)
    J_log = np.log1p(J)
    return (J_log - v_mean) / (v_std + 1e-12)


def value_inverse_transform(z: np.ndarray, v_mean: np.ndarray, v_std: np.ndarray) -> np.ndarray:
    """standardized log1p space -> raw J"""
    z = np.asarray(z, dtype=np.float64)
    J_log = z * (v_std + 1e-12) + v_mean
    return np.expm1(J_log)


# ---------------------------
# Metrics
# ---------------------------

def mse(a: np.ndarray, b: np.ndarray) -> float:
    d = (a - b).astype(np.float64)
    return float(np.mean(d * d))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def pearsonr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).reshape(-1)
    b = b.astype(np.float64).reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return float("nan")
    return float((a @ b) / denom)


@torch.no_grad()
def infer(
    pnet,
    vnet,
    X: np.ndarray,
    device: str,
    batch: int = 4096,
) -> Tuple[np.ndarray, np.ndarray]:
    pnet.eval()
    vnet.eval()

    X_t = torch.from_numpy(X.astype(np.float32))
    N = X_t.shape[0]

    pi_out = []
    v_out = []

    for i in range(0, N, batch):
        xb = X_t[i:i+batch].to(device)
        pi_pred = pnet(xb).detach().cpu().numpy()
        v_pred = vnet(xb).detach().cpu().numpy()
        pi_out.append(pi_pred)
        v_out.append(v_pred)

    return np.vstack(pi_out), np.vstack(v_out)


def maybe_make_plots(
    out_dir: Path,
    y_true_pi: np.ndarray,
    y_pred_pi: np.ndarray,
    y_true_J: np.ndarray,
    y_pred_J: np.ndarray,
    max_points: int = 5000,
) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    # Downsample for speed/legibility
    N = y_true_pi.shape[0]
    if N > max_points:
        idx = np.random.default_rng(0).choice(N, size=max_points, replace=False)
    else:
        idx = np.arange(N)

    # Policy scatter
    for d, name in enumerate(["theta0", "theta1"]):
        plt.figure()
        plt.scatter(y_true_pi[idx, d], y_pred_pi[idx, d], s=6)
        plt.xlabel(f"true {name}")
        plt.ylabel(f"pred {name}")
        plt.title(f"Policy scatter: {name}")
        plt.grid(True, alpha=0.3)
        plt.savefig(out_dir / f"policy_scatter_{name}.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Value scatter (raw J)
    plt.figure()
    plt.scatter(y_true_J[idx], y_pred_J[idx], s=6)
    plt.xlabel("true J*")
    plt.ylabel("pred J*")
    plt.title("Value scatter (raw cost)")
    plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / "value_scatter_raw.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Value scatter (log scale)
    plt.figure()
    plt.scatter(np.log1p(y_true_J[idx]), np.log1p(y_pred_J[idx]), s=6)
    plt.xlabel("true log1p(J*)")
    plt.ylabel("pred log1p(J*)")
    plt.title("Value scatter (log1p)")
    plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / "value_scatter_log1p.png", dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="pond4_lao_offline_data.npz", help="Path to dataset .npz (split_and_save_npz output)")
    ap.add_argument("--ckpt", type=str, default="saved_models/LAO_nets/lao_models.pt", help="Path to trained checkpoint (.pt)")
    ap.add_argument("--device", type=str, default="cuda", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--plots", action="store_true", help="Write scatter plots to --plot_dir")
    ap.add_argument("--plot_dir", type=str, default="eval_plots", help="Directory to save plots if --plots")
    ap.add_argument("--show_samples", type=int, default=8, help="Print a few (true/pred) samples")
    ap.add_argument('--hidden', type=int, default=128, help='Hidden size of the model')
    args = ap.parse_args()

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device != "cuda" else "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    pack = load_npz(args.data)
    ckpt = load_ckpt(args.ckpt, map_location=device)

    # dataset
    X_val = pack["X_val"].astype(np.float32)
    Ypi_val = pack["Ypi_val"].astype(np.float32)
    YV_val = pack["YV_val"].astype(np.float64)

    # filter non-finite YV (consistent with training)
    finite = np.isfinite(YV_val).reshape(-1)
    X_val = X_val[finite]
    Ypi_val = Ypi_val[finite]
    YV_val = YV_val[finite]

    # checkpoint params
    hidden = args.hidden
    v_mean = np.asarray(ckpt.get("v_mean", [[0.0]]), dtype=np.float64)
    v_std = np.asarray(ckpt.get("v_std", [[1.0]]), dtype=np.float64)
    v_transform = ckpt.get("value_transform", "none")

    # models
    if ("real_case" in args.data) or ("large" in args.ckpt):
        from train_lao_nets import PolicyNet_real, ValueNet_real
        pnet = PolicyNet_real(in_dim=X_val.shape[1], hidden=hidden).to(device)
        vnet = ValueNet_real(in_dim=X_val.shape[1], hidden=hidden).to(device)
    else:
        from train_lao_nets import PolicyNet, ValueNet
        pnet = PolicyNet(in_dim=X_val.shape[1], hidden=hidden).to(device)
        vnet = ValueNet(in_dim=X_val.shape[1], hidden=hidden).to(device)

    pnet.load_state_dict(ckpt["policy_state_dict"])
    vnet.load_state_dict(ckpt["value_state_dict"])

    # inference
    pi_pred, v_pred_z = infer(pnet, vnet, X_val, device=device, batch=args.batch)
    # v_pred_z is what the network outputs (should be z-scored log1p target if you trained that way)
    # compute value targets for comparison
    if v_transform == "log1p+zscore":
        y_true_z = value_forward_transform(YV_val.reshape(-1, 1), v_mean, v_std)
        y_pred_z = v_pred_z
        # inverse to raw J
        y_pred_J = value_inverse_transform(y_pred_z, v_mean, v_std).reshape(-1, 1)
    else:
        # assume raw J training
        y_true_z = YV_val.reshape(-1, 1)
        y_pred_z = v_pred_z
        y_pred_J = v_pred_z

    # policy metrics
    pi_mse = mse(pi_pred, Ypi_val)
    pi_mae = mae(pi_pred, Ypi_val)

    # per-dim
    pi_mse0 = mse(pi_pred[:, 0], Ypi_val[:, 0])
    pi_mse1 = mse(pi_pred[:, 1], Ypi_val[:, 1])
    pi_mae0 = mae(pi_pred[:, 0], Ypi_val[:, 0])
    pi_mae1 = mae(pi_pred[:, 1], Ypi_val[:, 1])

    # value metrics (transformed)
    v_mse_z = mse(y_pred_z, y_true_z)
    v_mae_z = mae(y_pred_z, y_true_z)

    # value metrics (raw)
    v_mse_J = mse(y_pred_J, YV_val.reshape(-1, 1))
    v_mae_J = mae(y_pred_J, YV_val.reshape(-1, 1))
    v_corr = pearsonr(y_pred_J.reshape(-1), YV_val.reshape(-1))

    print("\n=== Evaluation on validation set ===")
    print(f"N_val (finite) = {X_val.shape[0]}")
    print("\n[Policy]")
    print(f"  MSE  = {pi_mse:.6g}   MAE  = {pi_mae:.6g}")
    print(f"  theta0: MSE={pi_mse0:.6g} MAE={pi_mae0:.6g}")
    print(f"  theta1: MSE={pi_mse1:.6g} MAE={pi_mae1:.6g}")

    print("\n[Value]")
    if v_transform == "log1p+zscore":
        print("  transform = log1p+zscore (network predicts z)")
        print(f"  z-space:   MSE={v_mse_z:.6g}   MAE={v_mae_z:.6g}")
        print(f"  raw J*:    MSE={v_mse_J:.6g}   MAE={v_mae_J:.6g}   corr={v_corr:.4f}")
    else:
        print("  transform = none/raw (network predicts raw J*)")
        print(f"  raw J*:    MSE={v_mse_J:.6g}   MAE={v_mae_J:.6g}   corr={v_corr:.4f}")

    # show a few samples
    k = max(0, int(args.show_samples))
    if k > 0:
        k = min(k, X_val.shape[0])
        rng = np.random.default_rng(0)
        idx = rng.choice(X_val.shape[0], size=k, replace=False) if X_val.shape[0] > k else np.arange(k)
        print("\n[Samples] (true -> pred)")
        for i in idx:
            t0, t1 = Ypi_val[i]
            p0, p1 = pi_pred[i]
            jt = float(YV_val[i])
            jp = float(y_pred_J[i])
            print(f"  theta: ({t0:.3f},{t1:.3f}) -> ({p0:.3f},{p1:.3f}) | J*: {jt:.3e} -> {jp:.3e}")

    # plots
    if args.plots:
        out_dir = Path(args.plot_dir)
        maybe_make_plots(out_dir, Ypi_val, pi_pred, YV_val.reshape(-1), y_pred_J.reshape(-1))
        print(f"\nSaved plots to: {out_dir.resolve()}")

    print("\nDone.")


if __name__ == "__main__":
    main()
