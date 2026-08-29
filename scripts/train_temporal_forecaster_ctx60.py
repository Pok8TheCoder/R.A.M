"""Train context=60/horizon-capable GRU forecasters for both schemas (v2 -> the
"Y" line, amt -> the "A" line), for the 1000-step kill-chain experiment.

Most individual captures are far shorter than 60+40=100 flows (mean ~92,
median lower, many classes only have short captures), so training windows are
built from LONG per-class chains: every train-split capture of a class is
concatenated end-to-end (fixed random order), and only windows that don't
cross a splice boundary between two different original captures are kept.
This gives every class enough contiguous length to train on, even though most
individual captures could never supply a single (60-context, 40-horizon) pair
on their own.
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

from scripts.train_temporal_forecaster import TemporalForecaster  # noqa: E402

DATA = ROOT / "data" / "processed" / "forecast_captures.pkl"
CKPT_DIR = ROOT / "models" / "checkpoints"

CONTEXT = 60
HORIZON = 40
EPOCHS = 100
BATCH_SIZE = 256
LR = 1e-3
STEP_PCTL = 99.0
SEED = 0


def build_class_chains(captures, split, key):
    """Concatenate same-class captures end-to-end. Returns (chain, boundaries)
    per class, where boundaries[i]=True means chain[i] is the FIRST row of a
    new original capture (so a window must not straddle it)."""
    by_class: dict[str, list[np.ndarray]] = {}
    for c in captures:
        if c["split"] != split:
            continue
        by_class.setdefault(c["cls"], []).append(c[key])
    rng = np.random.default_rng(SEED)
    chains = {}
    for cls, arrs in by_class.items():
        order = list(range(len(arrs)))
        rng.shuffle(order)
        pieces, bounds = [], []
        for oi in order:
            a = arrs[oi]
            pieces.append(a)
            b = np.zeros(len(a), dtype=bool)
            b[0] = True
            bounds.append(b)
        chains[cls] = (np.concatenate(pieces), np.concatenate(bounds))
    return chains


def gather_windows_no_crossing(chains, context, horizon):
    Xs, Ys = [], []
    need = context + horizon
    for cls, (chain, bounds) in chains.items():
        n = len(chain)
        if n < need + 1:
            continue
        # a window starting at position p (0-indexed, covering [p, p+need)) is
        # valid only if no boundary marker (other than possibly at p itself,
        # which is fine -- the FIRST row of a window CAN be a capture start)
        # falls in (p, p+need).
        for p in range(0, n - need):
            if bounds[p + 1:p + need].any():
                continue
            Xs.append(chain[p:p + context])
            Ys.append(chain[p + context])
    return np.stack(Xs), np.stack(Ys)


def train_one_schema(name: str, captures, key: str, device) -> dict:
    train_chains = build_class_chains(captures, "train", key)
    val_chains = build_class_chains(captures, "val", key)

    Xtr, Ytr = gather_windows_no_crossing(train_chains, CONTEXT, HORIZON)
    Xva, Yva = gather_windows_no_crossing(val_chains, CONTEXT, HORIZON)
    dim = Ytr.shape[-1]
    print(f"  windows: train={len(Xtr)} val={len(Xva)}  dim={dim}  context={CONTEXT}")
    print(f"  classes with usable chains (train): {sorted(train_chains.keys())}")

    sc = StandardScaler().fit(Xtr.reshape(-1, dim))
    scale_seq = lambda A: ((A - sc.mean_) / sc.scale_).astype(np.float32)  # noqa: E731
    scale_flat = lambda A: ((A - sc.mean_) / sc.scale_).astype(np.float32)  # noqa: E731

    Xtr_s, Ytr_s = scale_seq(Xtr), scale_flat(Ytr)
    Xva_s, Yva_s = scale_seq(Xva), scale_flat(Yva)

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
    fit_time = time.time() - t0
    print(f"  -> best val MSE {best_val:.4f}  ({fit_time:.1f}s)")

    tag = "v2" if key == "v2" else "amt"
    torch.save({
        "model": model.state_dict(), "dim": dim, "window": CONTEXT,
        "max_step": max_step, "scaler_mean": sc.mean_, "scaler_scale": sc.scale_,
    }, CKPT_DIR / f"forecast_{tag}_temporal_ctx60.pth")
    return {"val_mse": best_val, "fit_time": fit_time}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(DATA, "rb") as f:
        captures = pickle.load(f)
    for schema, key in (("YMT (v2, 64f)", "v2"), ("AMT (aryan, 82f)", "amt")):
        print(f"\n=== {schema} : GRU temporal forecaster, context=60/horizon=40 ===")
        train_one_schema(schema, captures, key, device)


if __name__ == "__main__":
    main()
