"""
train_lao_nets.py

- Loads LAO dataset saved as .npz (from split_and_save_npz)
- Trains:
    * PolicyNet: x=[h,c] -> (theta0, theta1) in [0,1]^2
    * ValueNet : x=[h,c] -> J* (scalar)

Usage:
  python train_lao_nets.py --data lao_dataset.npz --out lao_models.pt --epochs 80

Notes:
- Assumes X in the npz is already normalized using X_mean/X_std saved in the file
  (as produced by split_and_save_npz(normalize_X=True)).
- We still load X_mean/X_std and store them with the model checkpoint for inference.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception as e:
    raise ImportError("This script requires PyTorch. Install with: pip install torch") from e


# ---------------------------
# Models (2-layer MLP = 1 hidden layer)
# ---------------------------

class ValueNet(nn.Module):
    def __init__(self, in_dim: int = 2, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PolicyNet(nn.Module):
    def __init__(self, in_dim: int = 2, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
            nn.Sigmoid(),  # clamp to (0,1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



class ValueNet_real(nn.Module):
    def __init__(self, in_dim: int = 2, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PolicyNet_real(nn.Module):
    def __init__(self, in_dim: int = 2, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
            nn.Sigmoid(),  # clamp to (0,1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------
# IO helpers
# ---------------------------

def load_lao_npz(path: str | Path) -> Dict[str, np.ndarray]:
    data = np.load(Path(path), allow_pickle=False)
    required = [
        "X_train", "Ypi_train", "YV_train",
        "X_val", "Ypi_val", "YV_val",
        "X_mean", "X_std",
    ]
    for k in required:
        if k not in data:
            raise KeyError(f"Missing key '{k}' in npz. Found keys: {list(data.keys())}")
    pack = {k: data[k] for k in data.files}
    return pack


@dataclass
class TrainConfig:
    hidden: int
    epochs: int
    batch: int
    lr: float
    weight_decay: float
    grad_clip: float
    patience: int
    seed: int


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------
# Training loop
# ---------------------------

@torch.no_grad()
def evaluate(
    pnet: PolicyNet,
    vnet: ValueNet,
    loader: DataLoader,
    device: str,
) -> Tuple[float, float]:
    pnet.eval()
    vnet.eval()

    mse = nn.MSELoss(reduction="sum")
    pi_sum = 0.0
    v_sum = 0.0
    n = 0

    for xb, ypi, yv in loader:
        xb = xb.to(device)
        ypi = ypi.to(device)
        yv = yv.to(device)

        p_pred = pnet(xb)
        v_pred = vnet(xb)

        pi_sum += float(mse(p_pred, ypi).item())
        v_sum += float(mse(v_pred, yv).item())
        n += xb.shape[0]

    # return mean MSE
    return pi_sum / max(n, 1), v_sum / max(n, 1)


def train(
    pnet: PolicyNet,
    vnet: ValueNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    cfg: TrainConfig,
) -> Dict[str, float]:
    mse = nn.MSELoss()

    opt_p = torch.optim.Adam(pnet.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    opt_v = torch.optim.Adam(vnet.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for ep in range(1, cfg.epochs + 1):
        pnet.train()
        vnet.train()

        train_pi = 0.0
        train_v = 0.0
        n = 0

        for xb, ypi, yv in train_loader:
            xb = xb.to(device)
            ypi = ypi.to(device)
            yv = yv.to(device)

            # --- policy step ---
            opt_p.zero_grad(set_to_none=True)
            p_pred = pnet(xb)
            loss_p = mse(p_pred, ypi)
            loss_p.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(pnet.parameters(), cfg.grad_clip)
            opt_p.step()

            # --- value step ---
            opt_v.zero_grad(set_to_none=True)
            v_pred = vnet(xb)
            loss_v = mse(v_pred, yv)
            loss_v.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(vnet.parameters(), cfg.grad_clip)
            opt_v.step()

            train_pi += float(loss_p.item()) * xb.shape[0]
            train_v += float(loss_v.item()) * xb.shape[0]
            n += xb.shape[0]

        train_pi /= max(n, 1)
        train_v /= max(n, 1)

        val_pi, val_v = evaluate(pnet, vnet, val_loader, device)
        val_total = val_pi + val_v

        print(
            f"Epoch {ep:03d}/{cfg.epochs} | "
            f"train_pi={train_pi:.6g} train_v={train_v:.6g} | "
            f"val_pi={val_pi:.6g} val_v={val_v:.6g} | "
            f"val_total={val_total:.6g}"
        )

        # Early stopping on combined validation loss
        if val_total < best_val - 1e-10:
            best_val = val_total
            best_state = {
                "policy": {k: v.detach().cpu() for k, v in pnet.state_dict().items()},
                "value": {k: v.detach().cpu() for k, v in vnet.state_dict().items()},
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if cfg.patience > 0 and bad_epochs >= cfg.patience:
                print(f"Early stopping: no improvement for {cfg.patience} epochs.")
                break

    # Restore best weights
    if best_state is not None:
        pnet.load_state_dict(best_state["policy"])
        vnet.load_state_dict(best_state["value"])

    # final metrics
    val_pi, val_v = evaluate(pnet, vnet, val_loader, device)
    return {"best_val_total": best_val, "val_pi": val_pi, "val_v": val_v}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="lao_dataset.npz", help="Path to .npz dataset")
    ap.add_argument("--out", type=str, default="lao_models.pt", help="Output checkpoint path (.pt)")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--large_model", default=False,action="store_true", help="Use larger model architecture for real case")
    
    args = ap.parse_args()

    cfg = TrainConfig(
        hidden=args.hidden,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        patience=args.patience,
        seed=args.seed,
    )

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device != "cuda" else "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    set_seed(cfg.seed)

    pack = load_lao_npz(args.data)
    X_train = pack["X_train"].astype(np.float32)
    Ypi_train = pack["Ypi_train"].astype(np.float32)
    YV_train = pack["YV_train"].astype(np.float32)

    X_val = pack["X_val"].astype(np.float32)
    Ypi_val = pack["Ypi_val"].astype(np.float32)
    YV_val = pack["YV_val"].astype(np.float32)

    # 1) remove non-finite (inf/nan)
    finite_tr = np.isfinite(YV_train).reshape(-1)
    finite_va = np.isfinite(YV_val).reshape(-1)

    X_train = X_train[finite_tr]
    Ypi_train = Ypi_train[finite_tr]
    YV_train = YV_train[finite_tr]

    X_val = X_val[finite_va]
    Ypi_val = Ypi_val[finite_va]
    YV_val = YV_val[finite_va]

    # 2) log transform (recommended)
    YV_train_log = np.log1p(YV_train)
    YV_val_log   = np.log1p(YV_val)

    # 3) standardize value targets using TRAIN stats
    v_mean = YV_train_log.mean(axis=0, keepdims=True)
    v_std  = YV_train_log.std(axis=0, keepdims=True) + 1e-8

    YV_train_tgt = (YV_train_log - v_mean) / v_std
    YV_val_tgt   = (YV_val_log   - v_mean) / v_std


    # Torch datasets/loaders
    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(Ypi_train),
        torch.from_numpy(YV_train_tgt),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(Ypi_val),
        torch.from_numpy(YV_val_tgt),
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch, shuffle=False, drop_last=False)

    # Models]

    if ("real_case" in args.data) or args.large_model:
        pnet = PolicyNet_real(in_dim=X_train.shape[1], hidden=cfg.hidden).to(device)
        vnet = ValueNet_real(in_dim=X_train.shape[1], hidden=cfg.hidden).to(device)
    else:
        pnet = PolicyNet(in_dim=X_train.shape[1], hidden=cfg.hidden).to(device)
        vnet = ValueNet(in_dim=X_train.shape[1], hidden=cfg.hidden).to(device)

    metrics = train(pnet, vnet, train_loader, val_loader, device, cfg)

    # Save checkpoint with normalization stats for inference
    ckpt = {
        "policy_state_dict": pnet.state_dict(),
        "value_state_dict": vnet.state_dict(),
        "hidden": cfg.hidden,
        "x_mean": pack["X_mean"].astype(np.float32),
        "x_std": pack["X_std"].astype(np.float32),
        "metrics": metrics,
        "meta_repr": pack.get("meta_repr", b""),
        "v_mean": v_mean, "v_std": v_std,
        "value_transform": "log1p+zscore"
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)
    print(f"Saved checkpoint to: {out_path.resolve()}")
    print(f"Final val_pi={metrics['val_pi']:.6g}, val_v={metrics['val_v']:.6g}")

if __name__ == "__main__":
    main()
