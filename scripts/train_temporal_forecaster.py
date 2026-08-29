"""Train a genuinely TEMPORAL forecaster: a GRU that processes a sequence of
past states in order, plus the same bounded-delta trick as the Hybrid model.

Why this is different from everything trained so far:
  - The classification models (ZMT.01/YMT.01/AMT.01) are called "Temporal
    Transformer" and that name is earned -- they attend over a window of
    flow states to classify the *current* one.
  - The forecasting "AE-MLP" from train_forecast_models.py is NOT temporal
    despite living in a forecasting script: it maps a single state ->
    single state, with zero memory of anything before it.
  - The "Hybrid" from train_hybrid_forecaster.py is a step in the right
    direction (it sees a window of 3 past states) but it's still not
    sequence-aware: the window is flattened and concatenated into one big
    vector, so the model has to learn "position 1 in the vector is one
    timestep older than position 2" from scratch, with no structural prior
    that these are ordered in time, and no ability to use windows longer
    than 3 without linearly growing the input (and parameter count).

This script is the first ACTUALLY temporal forecaster: a GRU consumes the
sequence one state at a time, carrying a hidden state forward, so the order
and recency of information is baked into the architecture, not just the
data layout. Same bounded-delta output head as the Hybrid, so we isolate
"recurrence vs flat concatenation" as the one variable that changed.
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data" / "processed" / "forecast_captures.pkl"
CKPT_DIR = ROOT / "models" / "checkpoints"

WINDOW = 6          # GRU gets more history than the Hybrid's 3, cheaply, since
                     # recurrence doesn't grow the input width with window length
EPOCHS = 100
BATCH_SIZE = 256
LR = 1e-3
HIDDEN = 64
STEP_PCTL = 99.0
SEED = 0


class TemporalForecaster(nn.Module):
    """GRU over a state sequence -> bounded delta on top of the last state."""

    def __init__(self, dim: int, max_step: torch.Tensor):
        super().__init__()
        self.dim = dim
        self.register_buffer("max_step", max_step)
        self.gru = nn.GRU(input_size=dim, hidden_size=HIDDEN, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN), nn.GELU(),
            nn.Linear(HIDDEN, dim),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (B, WINDOW, dim), most recent state is seq[:, -1, :]."""
        _, h_n = self.gru(seq)
        raw_delta = self.head(h_n.squeeze(0))
        delta = torch.tanh(raw_delta) * self.max_step
        return seq[:, -1, :] + delta


def gather_sequences(captures, split, key, window):
    Xs, Ys = [], []
    for c in captures:
        if c["split"] != split:
            continue
        traj = c[key]
        if len(traj) < window + 1:
            continue
        for t in range(window - 1, len(traj) - 1):
            Xs.append(traj[t - window + 1:t + 1])  # (window, dim)
            Ys.append(traj[t + 1])
    return np.stack(Xs), np.stack(Ys)


def train_one_schema(name: str, captures, key: str, device) -> dict:
    Xtr, Ytr = gather_sequences(captures, "train", key, WINDOW)
    Xva, Yva = gather_sequences(captures, "val", key, WINDOW)
    Xte, Yte = gather_sequences(captures, "test", key, WINDOW)
    dim = Ytr.shape[-1]
    print(f"  seqs: train={len(Xtr)} val={len(Xva)} test={len(Xte)}  dim={dim}  window={WINDOW}")

    sc = StandardScaler().fit(Xtr.reshape(-1, dim))

    def scale_seq(A):
        return ((A - sc.mean_) / sc.scale_).astype(np.float32)

    def scale_flat(A):
        return ((A - sc.mean_) / sc.scale_).astype(np.float32)

    Xtr_s, Ytr_s = scale_seq(Xtr), scale_flat(Ytr)
    Xva_s, Yva_s = scale_seq(Xva), scale_flat(Yva)
    Xte_s, Yte_s = scale_seq(Xte), scale_flat(Yte)

    cur_tr = Xtr_s[:, -1, :]
    abs_delta = np.abs(Ytr_s - cur_tr)
    max_step = np.clip(np.percentile(abs_delta, STEP_PCTL, axis=0), 1e-3, None)

    torch.manual_seed(SEED)
    model = TemporalForecaster(dim, torch.tensor(max_step, dtype=torch.float32)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    crit = nn.MSELoss()

    Xt, Yt = torch.from_numpy(Xtr_s).to(device), torch.from_numpy(Ytr_s).to(device)
    Xv, Yv = torch.from_numpy(Xva_s).to(device), torch.from_numpy(Yva_s).to(device)
    n = len(Xt)

    best_val, best_state = float("inf"), None
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            opt.zero_grad()
            loss = crit(model(Xt[idx]), Yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = crit(model(Xv), Yv).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 25 == 0:
            print(f"    [{name}] epoch {ep+1:>3}/{EPOCHS}  val_mse={val_loss:.4f}  best={best_val:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_mse = crit(model(torch.from_numpy(Xte_s).to(device)),
                        torch.from_numpy(Yte_s).to(device)).item()
    fit_time = time.time() - t0
    print(f"  -> test MSE {test_mse:.4f}  ({fit_time:.1f}s)")

    tag = "v2" if key == "v2" else "amt"
    torch.save({
        "model": model.state_dict(), "dim": dim, "window": WINDOW,
        "max_step": max_step, "scaler_mean": sc.mean_, "scaler_scale": sc.scale_,
    }, CKPT_DIR / f"forecast_{tag}_temporal.pth")
    return {"test_mse": test_mse, "fit_time": fit_time}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(DATA, "rb") as f:
        captures = pickle.load(f)
    for schema, key in (("YMT (v2, 64f)", "v2"), ("AMT (aryan, 82f)", "amt")):
        print(f"\n=== {schema} : GRU temporal forecaster ===")
        train_one_schema(schema, captures, key, device)


if __name__ == "__main__":
    main()
