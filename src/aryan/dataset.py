"""Sequence dataset for ARY world-model training from ``data/aryan_splits/*.npz``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent.parent.parent
SPLITS_DIR = ROOT / "data" / "aryan_splits"
XMT_SPLITS_DIR = ROOT / "data" / "xmt_splits"
LOOKBACK = 20


def load_split(name: str, splits_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = splits_dir or SPLITS_DIR
    d = np.load(root / f"{name}.npz")
    return (
        d["states"].astype(np.float32),
        d["labels_binary"].astype(np.int64),
        d["labels_mitre"].astype(np.int64),
    )


def load_all_splits(splits_dir: Path | None = None) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return {split: load_split(split, splits_dir) for split in ("train", "val", "test")}


class AryanSequenceDataset(Dataset):
    """One sample = (lookback window, next state, binary label, mitre label)."""

    def __init__(
        self,
        states: np.ndarray,
        labels_binary: np.ndarray,
        labels_mitre: np.ndarray,
        lookback: int = LOOKBACK,
        col_idx: list[int] | None = None,
    ):
        self.states = states if col_idx is None else states[:, col_idx]
        self.labels_binary = labels_binary
        self.labels_mitre = labels_mitre
        self.lookback = lookback
        self.indices = list(range(lookback - 1, len(states) - 1))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        t = self.indices[i]
        seq = self.states[t - self.lookback + 1 : t + 1]
        nxt = self.states[t + 1]
        return (
            torch.from_numpy(seq),
            torch.from_numpy(nxt),
            torch.tensor(self.labels_binary[t + 1], dtype=torch.long),
            torch.tensor(self.labels_mitre[t + 1], dtype=torch.long),
        )


def fit_scaler(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = states.mean(axis=0)
    std = np.clip(states.std(axis=0), 1e-6, None)
    return mean.astype(np.float32), std.astype(np.float32)


def transform_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((states - mean) / std).astype(np.float32)
