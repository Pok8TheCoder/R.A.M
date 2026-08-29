"""Same GRU temporal forecaster as train_temporal_forecaster.py, but with a
20-step context window (matching the adaptive-memory experiment's spec:
"20 steps of context, predict next 20 steps"). v2 schema only, to keep the
adaptive-memory prototype focused.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.train_temporal_forecaster as base  # noqa: E402

CKPT_DIR = ROOT / "models" / "checkpoints"


def main():
    base.WINDOW = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(base.DATA, "rb") as f:
        captures = pickle.load(f)

    # train_one_schema always writes to forecast_v2_temporal.pth (window=6, used
    # by the v4 rollout results) -- back it up first so this doesn't clobber it.
    orig = CKPT_DIR / "forecast_v2_temporal.pth"
    backup = CKPT_DIR / "forecast_v2_temporal.pth.window6.bak"
    if orig.exists() and not backup.exists():
        backup.write_bytes(orig.read_bytes())

    print("=== YMT (v2, 64f) : GRU temporal forecaster, context=20 ===")
    res = base.train_one_schema("v2-ctx20", captures, "v2", device)

    dst = CKPT_DIR / "forecast_v2_temporal_ctx20.pth"
    orig.rename(dst)
    if backup.exists():
        backup.rename(orig)
    print(f"Saved -> {dst}  (test_mse={res['test_mse']:.4f})")
    print(f"Restored original window=6 checkpoint at -> {orig}")


if __name__ == "__main__":
    main()
