"""Stateful, streaming versions of the ARY+RAM and TimesFM+ARY-hybrid variants
compared in `scripts/ram_improve_eval.py` / `scripts/timesfm_ary_lab_eval.py`.

Those scripts evaluate over a fixed, already-captured array of states (batch).
The live adversarial lab (`scripts/live_attack_lab.py`) instead sees one new
242-d window at a time as traffic is captured in near-real-time, so the
per-step body of `run_variant()` is refactored here into two classes that can
be `.step()`-ped incrementally:

  StreamingARY(mode)              -- ary_base | ary_v8 | ary_v10
  StreamingTimesFMHybrid(ram_mode) -- classifier reuses ARY's frozen head,
                                       dynamics/forecast come from TimesFM;
                                       ram_mode is none | ungated | gated

Both preserve the exact numerical behavior of their batch counterparts,
including one important temporal subtlety: `run_variant()` writes each
step's classify-blend memory entry using the *next* step's revealed label
(`bank.add(key_t, true_bin_{t+1}, true_mit_{t+1})`), which is legitimate in
a batch replay (the whole label array is just sitting there) but must be
made explicit in a streaming setting, since label `t+1` genuinely isn't
known until window `t+1` actually arrives. Both classes therefore hold a
one-step "pending" slot: the key/prediction computed on `.step()` for
window t is only written into memory / used for TTT once `.step()` is
called again for window t+1 with that window's now-revealed ground truth.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from src.aryan.components import MultiTaskLoss
from src.aryan.world_model import TemporalTransformerWorldModel

ROOT = Path(__file__).resolve().parent.parent.parent
CKPT = ROOT / "models" / "checkpoints" / "aryan_world_model_best.pt"

# Exact hyperparameters from scripts/ram_improve_eval.py -- kept in lockstep
# so streaming variants numerically match the batch eval results already
# validated (V8/V10 F1 deltas etc).
CONTEXT = 20
ADAPT_EVERY = 20
ADAPT_STEPS = 3
ADAPT_LR = 3e-4
PULLBACK = 5e-3
BLEND_MAX_WEIGHT = 0.6
BLEND_FLOOR = 0.15
NUM_MITRE_STAGES = 7

MODE_SPECS: dict[str, dict[str, Any]] = {
    "ary_base": dict(adapt=False, loss_type="mse", use_memory=False, key_space="raw", knn_k=1),
    "ary_v8": dict(adapt=False, loss_type="mse", use_memory=True, key_space="raw", knn_k=1),
    "ary_v10": dict(adapt=True, loss_type="multitask", use_memory=True, key_space="hidden", knn_k=3),
}

RAM_MODES = ("none", "ungated", "gated")

SYSTEM_IDS = [
    "ary_base",
    "ary_v8",
    "ary_v10",
    "timesfm_ary_cls",
    "fmary_ram_ungated",
    "fmary_gated_ram",
]


def load_model(ckpt_path: Path = CKPT) -> TemporalTransformerWorldModel:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    d_state = ck["model_state_dict"]["embedding.proj.weight"].shape[1]
    model = TemporalTransformerWorldModel(d_state=d_state, d_model=256, n_layers=4, n_heads=8, lookback=CONTEXT)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model


def calibrate_match_thresh(keys: np.ndarray, labels: np.ndarray) -> float:
    """Same/diff-label NN distance calibration, identical to
    ram_improve_eval.py::calibrate_match_thresh (duplicated here so this
    module has no import-time dependency on scripts/)."""
    n = len(keys)
    if n < 10:
        return 30.0
    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=min(200, n), replace=False)
    same_d, diff_d = [], []
    for i in idx:
        d = np.linalg.norm(keys - keys[i][None, :], axis=1)
        for j in range(n):
            if j == i:
                continue
            (same_d if labels[j] == labels[i] else diff_d).append(d[j])
    if not same_d or not diff_d:
        return 30.0
    return float((np.percentile(same_d, 40) + np.percentile(diff_d, 10)) / 2)


def calibrate_thresholds(base_model: nn.Module, va_states: np.ndarray, va_bin_labels: np.ndarray) -> tuple[float, float]:
    """One-shot calibration of raw-window / hidden-state match thresholds on
    a labeled validation split (e.g. the aryan CIC-IDS val split), meant to
    be called once at orchestrator startup and passed into every streaming
    system's constructor."""
    val_raw_keys, val_hidden_keys = [], []
    for t in range(CONTEXT - 1, len(va_states) - 1):
        seq = va_states[t - CONTEXT + 1 : t + 1]
        out = infer(base_model, seq)
        val_raw_keys.append(seq.reshape(-1))
        val_hidden_keys.append(out["hidden"])
    val_raw_keys = np.stack(val_raw_keys)
    val_hidden_keys = np.stack(val_hidden_keys)
    val_bin = va_bin_labels[CONTEXT : len(va_states)]
    raw_thresh = calibrate_match_thresh(val_raw_keys, val_bin)
    hidden_thresh = calibrate_match_thresh(val_hidden_keys, val_bin)
    return raw_thresh, hidden_thresh


class MemoryBank:
    """Classify-blend episodic memory: nearest-neighbor lookup keyed by raw
    window or hidden state, voting on binary/MITRE labels. Identical
    mechanics to ram_improve_eval.py::MemoryBank."""

    def __init__(self):
        self.keys: list[np.ndarray] = []
        self.bin_labels: list[int] = []
        self.mit_labels: list[int] = []

    def add(self, key: np.ndarray, bin_label: int, mit_label: int) -> None:
        self.keys.append(key)
        self.bin_labels.append(int(bin_label))
        self.mit_labels.append(int(mit_label))

    def query(self, key: np.ndarray, k: int = 1) -> list[tuple[int, int, float]]:
        if not self.keys:
            return []
        M = np.stack(self.keys)
        d = np.linalg.norm(M - key[None, :], axis=1)
        idx = np.argsort(d)[:k]
        return [(self.bin_labels[i], self.mit_labels[i], float(d[i])) for i in idx]

    def __len__(self) -> int:
        return len(self.keys)


class TieredMemoryBank:
    """Two-tier classify-blend memory, designed to fix the dilution failure
    found empirically in the live lab: the original `MemoryBank` writes
    *every* window forever (benign or not), so once enough quiet history
    piles up, a real attack's k-NN vote gets swamped by irrelevant "boring
    normal" neighbors and never crosses the detection threshold even though
    the raw signal is clearly elevated.

    Tier 1 -- `baseline`: the first `baseline_capacity` full-context windows
    of the run (assumed benign warm-up), written once and then frozen. A
    fixed "what normal looks like" reference, never grows or evicts.

    Tier 2 -- `dynamic`: starts *empty*. Post-baseline windows are only
    ever written here if they're suspicious (raw pre-blend P(attack) >=
    `suspicious_thresh`) or a confirmed attack (`true_bin == 1`) -- plain
    benign windows are simply never stored, so the notebook stops
    accumulating noise once past the initial baseline. Starts at
    `dynamic_capacity_pre` slots; the first time a *confirmed* attack is
    written, capacity grows once and permanently to `dynamic_capacity_post`.
    FIFO eviction once at capacity (oldest suspicious/attack entry drops).

    `.query()`/`__len__` duck-type `MemoryBank` so `_classify_blend` and
    `StreamingARY` need no changes to use this in place of the flat bank.
    """

    def __init__(
        self,
        baseline_capacity: int = 20,
        dynamic_capacity_pre: int = 20,
        dynamic_capacity_post: int = 40,
        suspicious_thresh: float = 0.01,
    ):
        self.baseline_capacity = baseline_capacity
        self.dynamic_capacity_pre = dynamic_capacity_pre
        self.dynamic_capacity_post = dynamic_capacity_post
        self.suspicious_thresh = suspicious_thresh

        self.baseline_keys: list[np.ndarray] = []
        self.baseline_bin: list[int] = []
        self.baseline_mit: list[int] = []

        self.dynamic_keys: list[np.ndarray] = []
        self.dynamic_bin: list[int] = []
        self.dynamic_mit: list[int] = []
        self._dynamic_capacity = dynamic_capacity_pre
        self._grown = False

    @property
    def baseline_full(self) -> bool:
        return len(self.baseline_keys) >= self.baseline_capacity

    def add_baseline(self, key: np.ndarray, bin_label: int, mit_label: int) -> None:
        if self.baseline_full:
            return
        self.baseline_keys.append(key)
        self.baseline_bin.append(int(bin_label))
        self.baseline_mit.append(int(mit_label))

    def maybe_add_dynamic(self, key: np.ndarray, bin_label: int, mit_label: int, p_att_raw: float) -> bool:
        """Writes only if the window is suspicious or a confirmed attack;
        returns True iff it actually wrote (useful for diagnostics)."""
        is_attack = bin_label == 1
        is_suspicious = is_attack or p_att_raw >= self.suspicious_thresh
        if not is_suspicious:
            return False

        if is_attack and not self._grown:
            self._dynamic_capacity = self.dynamic_capacity_post
            self._grown = True

        if len(self.dynamic_keys) >= self._dynamic_capacity:
            self.dynamic_keys.pop(0)
            self.dynamic_bin.pop(0)
            self.dynamic_mit.pop(0)

        self.dynamic_keys.append(key)
        self.dynamic_bin.append(int(bin_label))
        self.dynamic_mit.append(int(mit_label))
        return True

    def query(self, key: np.ndarray, k: int = 1) -> list[tuple[int, int, float]]:
        keys = self.baseline_keys + self.dynamic_keys
        bins = self.baseline_bin + self.dynamic_bin
        mits = self.baseline_mit + self.dynamic_mit
        if not keys:
            return []
        M = np.stack(keys)
        d = np.linalg.norm(M - key[None, :], axis=1)
        idx = np.argsort(d)[:k]
        return [(bins[i], mits[i], float(d[i])) for i in idx]

    def __len__(self) -> int:
        return len(self.baseline_keys) + len(self.dynamic_keys)


class RAMXMemoryBank:
    """RAMX_V.01 -- rolling benign baseline + gated dynamic tier.

    Baseline (20 slots, four quintiles of 5):
      - On a 100%-regular (benign) stream, new entries land in quintile 4
        (1-indexed slots 16-20; indices 15-19).
      - Every ``rotation_interval`` full-context benign steps, if the stream
        is still benign, cascade one quintile down:
          Q1 <- Q2, Q2 <- Q3, Q3 <- Q4, Q4 <- empty (fresh writes).
        After four rotations the whole 20-slot baseline has been refreshed;
        nothing stays frozen indefinitely.

    Dynamic tier (slots 21-60 once an attack is seen):
      - Starts empty (capacity 20). Only suspicious windows
        (raw P(attack) >= ``suspicious_thresh``) or confirmed attacks
        (``true_bin == 1``) are written. Capacity grows to 40 on first
        confirmed attack. FIFO eviction at capacity.

    Duck-types ``MemoryBank`` for ``_classify_blend``.
    """

    QUINTILE_SIZE = 5
    N_QUINTILES = 4
    BASELINE_CAPACITY = 20  # 4 x 5

    def __init__(
        self,
        rotation_interval: int = 100,
        dynamic_capacity_pre: int = 20,
        dynamic_capacity_post: int = 40,
        suspicious_thresh: float = 0.01,
    ):
        self.rotation_interval = rotation_interval
        self.dynamic_capacity_pre = dynamic_capacity_pre
        self.dynamic_capacity_post = dynamic_capacity_post
        self.suspicious_thresh = suspicious_thresh

        self._slots: list[Optional[tuple[np.ndarray, int, int]]] = [None] * self.BASELINE_CAPACITY
        self._active_quintile = self.N_QUINTILES - 1  # start at Q4 (16-20)
        self._quintile_write_idx = 0  # next slot within active quintile (0..4)
        self._benign_steps_since_rotation = 0
        self._rotation_count = 0

        self.dynamic_keys: list[np.ndarray] = []
        self.dynamic_bin: list[int] = []
        self.dynamic_mit: list[int] = []
        self._dynamic_capacity = dynamic_capacity_pre
        self._attack_seen = False

    def _quintile_range(self, q: int) -> range:
        start = q * self.QUINTILE_SIZE
        return range(start, start + self.QUINTILE_SIZE)

    def _occupied_baseline_count(self) -> int:
        return sum(1 for s in self._slots if s is not None)

    def _rotate_baseline(self) -> None:
        """Cascade Q4->Q3->Q2->Q1, clear Q4 for fresh benign writes."""
        for q in range(self.N_QUINTILES - 1):
            src_start = (q + 1) * self.QUINTILE_SIZE
            dst_start = q * self.QUINTILE_SIZE
            for i in range(self.QUINTILE_SIZE):
                self._slots[dst_start + i] = self._slots[src_start + i]
        for i in self._quintile_range(self.N_QUINTILES - 1):
            self._slots[i] = None
        self._quintile_write_idx = 0
        self._benign_steps_since_rotation = 0
        self._rotation_count += 1

    def _write_baseline_quintile(self, key: np.ndarray, bin_label: int, mit_label: int) -> None:
        q = self._active_quintile
        base = q * self.QUINTILE_SIZE
        slot_idx = base + (self._quintile_write_idx % self.QUINTILE_SIZE)
        self._slots[slot_idx] = (key, int(bin_label), int(mit_label))
        self._quintile_write_idx += 1

    def on_benign_step(self, key: np.ndarray, bin_label: int, mit_label: int) -> None:
        """Record one confirmed-benign full-context window into baseline."""
        self._write_baseline_quintile(key, bin_label, mit_label)
        self._benign_steps_since_rotation += 1
        if self._benign_steps_since_rotation >= self.rotation_interval:
            self._rotate_baseline()

    def maybe_add_dynamic(self, key: np.ndarray, bin_label: int, mit_label: int, p_att_raw: float) -> bool:
        is_attack = bin_label == 1
        is_suspicious = is_attack or p_att_raw >= self.suspicious_thresh
        if not is_suspicious:
            return False
        if is_attack and not self._attack_seen:
            self._dynamic_capacity = self.dynamic_capacity_post
            self._attack_seen = True
        if len(self.dynamic_keys) >= self._dynamic_capacity:
            self.dynamic_keys.pop(0)
            self.dynamic_bin.pop(0)
            self.dynamic_mit.pop(0)
        self.dynamic_keys.append(key)
        self.dynamic_bin.append(int(bin_label))
        self.dynamic_mit.append(int(mit_label))
        return True

    def ingest(
        self,
        key: np.ndarray,
        bin_label: int,
        mit_label: int,
        p_att_raw: float,
    ) -> None:
        """Route one deferred write: suspicious/attack -> dynamic; else benign baseline."""
        is_suspicious_or_attack = bin_label == 1 or p_att_raw >= self.suspicious_thresh
        if is_suspicious_or_attack:
            self.maybe_add_dynamic(key, bin_label, mit_label, p_att_raw)
        elif bin_label == 0:
            self.on_benign_step(key, bin_label, mit_label)

    def query(self, key: np.ndarray, k: int = 1) -> list[tuple[int, int, float]]:
        keys, bins, mits = [], [], []
        for slot in self._slots:
            if slot is not None:
                keys.append(slot[0])
                bins.append(slot[1])
                mits.append(slot[2])
        keys.extend(self.dynamic_keys)
        bins.extend(self.dynamic_bin)
        mits.extend(self.dynamic_mit)
        if not keys:
            return []
        M = np.stack(keys)
        d = np.linalg.norm(M - key[None, :], axis=1)
        idx = np.argsort(d)[:k]
        return [(bins[i], mits[i], float(d[i])) for i in idx]

    def __len__(self) -> int:
        return self._occupied_baseline_count() + len(self.dynamic_keys)

    @property
    def baseline_occupied(self) -> int:
        return self._occupied_baseline_count()

    @property
    def rotation_count(self) -> int:
        return self._rotation_count


@torch.no_grad()
def infer(model: nn.Module, seq: np.ndarray) -> dict:
    x = torch.from_numpy(seq[None].astype(np.float32))
    out = model(x)
    return {
        "p_att": torch.softmax(out["pred_binary"], -1)[0, 1].item(),
        "p_mit": torch.softmax(out["pred_mitre"], -1)[0].numpy(),
        "hidden": out["hidden"][0].numpy(),
        "pred_state": out["pred_state_mean"][0].numpy(),
    }


def _classify_blend(
    bank: MemoryBank,
    key: np.ndarray,
    p_att: float,
    p_mit: np.ndarray,
    knn_k: int,
    match_thresh: float,
) -> tuple[float, np.ndarray]:
    """Shared classify-blend math used by both StreamingARY (V8/V10) and
    StreamingTimesFMHybrid's RAM. Identical to the blend block inside
    ram_improve_eval.py::run_variant."""
    if len(bank) == 0:
        return p_att, p_mit
    matches = bank.query(key, k=knn_k)
    close = [m for m in matches if m[2] < match_thresh]
    if not close:
        return p_att, p_mit
    weights = np.array([max(0.0, 1.0 - d / match_thresh) for _, _, d in close])
    weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(close)) / len(close)
    w_total = min(BLEND_MAX_WEIGHT, float(np.mean(weights))) * 0.7 + BLEND_FLOOR
    vote_bin = float(np.sum([w * b for (b, _, _), w in zip(close, weights)]))
    p_att_out = (1 - w_total) * p_att + w_total * vote_bin
    mit_onehot = np.zeros(len(p_mit), dtype=np.float32)
    for (_, m, _), w in zip(close, weights):
        mit_onehot[m] += w
    p_mit_out = (1 - w_total) * p_mit + w_total * mit_onehot
    return p_att_out, p_mit_out


class StreamingARY:
    """Stateful ARY.01 classifier/dynamics model, one of `ary_base` / `ary_v8`
    / `ary_v10` (exact kwargs from ram_improve_eval.py's V0/V8_frozen_mem_raw_k1/
    V10_mt_mem_hidden_k3). Feed one 242-d window per `.step()` call.

    Ground truth (`true_bin`/`true_mit`) for the window just passed to
    `.step()` should be supplied on the *next* call (once it's known) --
    see module docstring for why the memory write / TTT sample are deferred
    by one step to match the batch scripts' temporal semantics exactly.
    """

    def __init__(
        self,
        mode: str,
        base_model: Optional[nn.Module] = None,
        raw_thresh: Optional[float] = None,
        hidden_thresh: Optional[float] = None,
    ):
        if mode not in MODE_SPECS:
            raise ValueError(f"Unknown StreamingARY mode: {mode}")
        self.mode = mode
        spec = MODE_SPECS[mode]
        self.adapt: bool = spec["adapt"]
        self.loss_type: str = spec["loss_type"]
        self.use_memory: bool = spec["use_memory"]
        self.key_space: str = spec["key_space"]
        self.knn_k: int = spec["knn_k"]

        base = base_model if base_model is not None else load_model()
        self.online = copy.deepcopy(base)
        self.base_params = [p.clone().detach() for p in base.parameters()]
        self.opt = torch.optim.SGD(self.online.parameters(), lr=ADAPT_LR) if self.adapt else None
        self._mse = nn.MSELoss()
        self._mt_loss = MultiTaskLoss(lambda_dynamics=0.5, lambda_infiltration=1.2, lambda_mitre=1.0)

        self.bank = MemoryBank()
        self.match_thresh = (hidden_thresh if self.key_space == "hidden" else raw_thresh) or 30.0

        self.buffer: list[np.ndarray] = []
        self._adapt_buf: dict[str, list] = {"seqs": [], "next": [], "tb": [], "tm": []}
        self._pending: Optional[dict] = None
        self.last_dynamics_mse: Optional[float] = None
        self.n_steps = 0

    def _padded_seq(self) -> np.ndarray:
        window = self.buffer[-CONTEXT:]
        if len(window) < CONTEXT:
            window = [window[0]] * (CONTEXT - len(window)) + window
        return np.stack(window).astype(np.float32)

    def step(
        self,
        state: np.ndarray,
        true_bin: Optional[int] = None,
        true_mit: Optional[int] = None,
    ) -> dict:
        state = np.asarray(state, dtype=np.float32)
        self.n_steps += 1

        if self._pending is not None:
            self._finalize_pending(state, true_bin, true_mit)

        self.buffer.append(state)
        if len(self.buffer) > CONTEXT:
            self.buffer.pop(0)
        full_context = len(self.buffer) >= CONTEXT
        seq = self._padded_seq()

        self.online.eval()
        out = infer(self.online, seq)
        p_att, p_mit, hidden, pred_state = out["p_att"], out["p_mit"], out["hidden"], out["pred_state"]
        key = hidden if self.key_space == "hidden" else seq.reshape(-1)

        if self.use_memory:
            p_att, p_mit = _classify_blend(self.bank, key, p_att, p_mit, self.knn_k, self.match_thresh)

        self._pending = {"key": key, "seq": seq, "pred_state": pred_state, "full_context": full_context}

        return {
            "p_att": float(p_att),
            "p_mit": p_mit,
            "pred_state": pred_state,
            "hidden": hidden,
        }

    def _finalize_pending(self, state: np.ndarray, true_bin: Optional[int], true_mit: Optional[int]) -> None:
        pend = self._pending
        self._pending = None
        # Lab-captured traffic can be far OOD relative to the CIC-IDS training
        # distribution (unnormalized live packet/byte counts vs. aggregated
        # training scale), which can blow up the dynamics MSE numerically;
        # clamp for JSON/plot sanity (same fix applied in timesfm_ary_lab_eval.py).
        raw_mse = float(np.mean((pend["pred_state"] - state) ** 2))
        self.last_dynamics_mse = float(min(raw_mse, 1e12)) if np.isfinite(raw_mse) else 1e12
        if true_bin is None or not pend["full_context"]:
            # Matches the batch scripts exactly: they never write memory/TTT
            # samples keyed on a padded (< CONTEXT real windows) context,
            # since their loop only ever starts at t=CONTEXT-1 with a fully
            # real, unpadded window already available.
            return
        tb = int(true_bin)
        tm = int(true_mit) if true_mit is not None else 0

        if self.use_memory:
            self.bank.add(pend["key"], tb, tm)

        if self.adapt:
            buf = self._adapt_buf
            buf["seqs"].append(pend["seq"])
            buf["next"].append(state)
            buf["tb"].append(tb)
            buf["tm"].append(tm)
            if len(buf["seqs"]) >= ADAPT_EVERY:
                self._run_ttt()

    def _run_ttt(self) -> None:
        buf = self._adapt_buf
        seqs_t = torch.from_numpy(np.stack(buf["seqs"]).astype(np.float32))
        next_t = torch.from_numpy(np.stack(buf["next"]).astype(np.float32))
        tb_t = torch.tensor(buf["tb"], dtype=torch.long)
        tm_t = torch.tensor(buf["tm"], dtype=torch.long)

        self.online.train()
        for _ in range(ADAPT_STEPS):
            self.opt.zero_grad()
            out_a = self.online(seqs_t)
            if self.loss_type == "mse":
                loss = self._mse(out_a["pred_state_mean"], next_t)
            else:
                ld = self._mt_loss(
                    out_a["pred_state_mean"], out_a["pred_state_logvar"], next_t,
                    out_a["pred_binary"], tb_t, out_a["pred_mitre"], tm_t,
                )
                loss = ld["total"]
            reg = sum(((p - b) ** 2).sum() for p, b in zip(self.online.parameters(), self.base_params))
            (loss + PULLBACK * reg).backward()
            self.opt.step()
        self.online.eval()
        buf["seqs"], buf["next"], buf["tb"], buf["tm"] = [], [], [], []

    @torch.no_grad()
    def forecast(self, horizon: int = 100) -> np.ndarray:
        """Autoregressive `horizon`-step rollout from the current buffer.
        Read-only (does not mutate streaming state / online weights):
        used for the viewer's "did we see it coming" forecast panel, not
        for live win/loss scoring (per plan, that stays on 1-step p_att)."""
        self.online.eval()
        buf = list(self._padded_seq())
        preds = []
        for _ in range(horizon):
            x = torch.from_numpy(np.stack(buf).astype(np.float32)).unsqueeze(0)
            out = self.online(x)
            nxt = out["pred_state_mean"].squeeze(0).numpy()
            preds.append(nxt)
            buf = buf[1:] + [nxt]
        return np.stack(preds)


class StreamingARYTieredV10(StreamingARY):
    """`ary_v10` (TTT + hidden-key/k=3 classify-blend), but with
    `TieredMemoryBank` in place of V10's normal flat, unbounded `MemoryBank`.

    Only overrides `step`/`_finalize_pending` to (a) stash the raw
    pre-blend P(attack) so the "is this suspicious" gate has something to
    check, and (b) route memory writes through the tiered baseline/dynamic
    policy instead of `self.bank.add()` unconditionally. TTT and the
    classify-blend math itself (`_classify_blend`) are untouched.
    """

    def __init__(
        self,
        base_model: Optional[nn.Module] = None,
        hidden_thresh: Optional[float] = None,
        baseline_capacity: int = 20,
        dynamic_capacity_pre: int = 20,
        dynamic_capacity_post: int = 40,
        suspicious_thresh: float = 0.01,
    ):
        super().__init__("ary_v10", base_model=base_model, hidden_thresh=hidden_thresh)
        self.bank = TieredMemoryBank(
            baseline_capacity=baseline_capacity,
            dynamic_capacity_pre=dynamic_capacity_pre,
            dynamic_capacity_post=dynamic_capacity_post,
            suspicious_thresh=suspicious_thresh,
        )

    def step(
        self,
        state: np.ndarray,
        true_bin: Optional[int] = None,
        true_mit: Optional[int] = None,
    ) -> dict:
        state = np.asarray(state, dtype=np.float32)
        self.n_steps += 1

        if self._pending is not None:
            self._finalize_pending(state, true_bin, true_mit)

        self.buffer.append(state)
        if len(self.buffer) > CONTEXT:
            self.buffer.pop(0)
        full_context = len(self.buffer) >= CONTEXT
        seq = self._padded_seq()

        self.online.eval()
        out = infer(self.online, seq)
        p_att_raw, p_mit_raw, hidden, pred_state = out["p_att"], out["p_mit"], out["hidden"], out["pred_state"]
        key = hidden if self.key_space == "hidden" else seq.reshape(-1)

        p_att, p_mit = p_att_raw, p_mit_raw
        if self.use_memory:
            p_att, p_mit = _classify_blend(self.bank, key, p_att_raw, p_mit_raw, self.knn_k, self.match_thresh)

        self._pending = {
            "key": key, "seq": seq, "pred_state": pred_state,
            "full_context": full_context, "p_att_raw": p_att_raw,
        }

        return {"p_att": float(p_att), "p_mit": p_mit, "pred_state": pred_state, "hidden": hidden}

    def _finalize_pending(self, state: np.ndarray, true_bin: Optional[int], true_mit: Optional[int]) -> None:
        pend = self._pending
        self._pending = None
        raw_mse = float(np.mean((pend["pred_state"] - state) ** 2))
        self.last_dynamics_mse = float(min(raw_mse, 1e12)) if np.isfinite(raw_mse) else 1e12
        if true_bin is None or not pend["full_context"]:
            return
        tb = int(true_bin)
        tm = int(true_mit) if true_mit is not None else 0

        if self.use_memory:
            bank: TieredMemoryBank = self.bank  # type: ignore[assignment]
            is_suspicious_or_attack = tb == 1 or pend["p_att_raw"] >= bank.suspicious_thresh
            if is_suspicious_or_attack:
                # Suspicious/attack windows always go to the dynamic tier,
                # even if this happens to be one of the round's first
                # full-context windows (e.g. warmup_windows == CONTEXT means
                # the attack can start the instant full_context flips true --
                # it must never get swept into the "benign baseline" tier).
                bank.maybe_add_dynamic(pend["key"], tb, tm, pend["p_att_raw"])
            elif not bank.baseline_full:
                bank.add_baseline(pend["key"], tb, tm)
            # else: plain benign after baseline is full -- deliberately
            # dropped, not stored anywhere (that's the "left empty until
            # suspicious activity" part of the design).

        if self.adapt:
            buf = self._adapt_buf
            buf["seqs"].append(pend["seq"])
            buf["next"].append(state)
            buf["tb"].append(tb)
            buf["tm"].append(tm)
            if len(buf["seqs"]) >= ADAPT_EVERY:
                self._run_ttt()


class StreamingARYRamxV01(StreamingARY):
    """ARY + RAMX_V.01: V10 backbone (TTT + hidden-key k=3 classify-blend)
    with ``RAMXMemoryBank`` rolling benign baseline instead of flat memory."""

    def __init__(
        self,
        base_model: Optional[nn.Module] = None,
        hidden_thresh: Optional[float] = None,
        rotation_interval: int = 100,
        dynamic_capacity_pre: int = 20,
        dynamic_capacity_post: int = 40,
        suspicious_thresh: float = 0.01,
    ):
        super().__init__("ary_v10", base_model=base_model, hidden_thresh=hidden_thresh)
        self.bank = RAMXMemoryBank(
            rotation_interval=rotation_interval,
            dynamic_capacity_pre=dynamic_capacity_pre,
            dynamic_capacity_post=dynamic_capacity_post,
            suspicious_thresh=suspicious_thresh,
        )

    def step(
        self,
        state: np.ndarray,
        true_bin: Optional[int] = None,
        true_mit: Optional[int] = None,
    ) -> dict:
        state = np.asarray(state, dtype=np.float32)
        self.n_steps += 1

        if self._pending is not None:
            self._finalize_pending(state, true_bin, true_mit)

        self.buffer.append(state)
        if len(self.buffer) > CONTEXT:
            self.buffer.pop(0)
        full_context = len(self.buffer) >= CONTEXT
        seq = self._padded_seq()

        self.online.eval()
        out = infer(self.online, seq)
        p_att_raw, p_mit_raw, hidden, pred_state = out["p_att"], out["p_mit"], out["hidden"], out["pred_state"]
        key = hidden if self.key_space == "hidden" else seq.reshape(-1)

        p_att, p_mit = p_att_raw, p_mit_raw
        if self.use_memory:
            p_att, p_mit = _classify_blend(self.bank, key, p_att_raw, p_mit_raw, self.knn_k, self.match_thresh)

        self._pending = {
            "key": key, "seq": seq, "pred_state": pred_state,
            "full_context": full_context, "p_att_raw": p_att_raw,
        }

        return {"p_att": float(p_att), "p_mit": p_mit, "pred_state": pred_state, "hidden": hidden}

    def _finalize_pending(self, state: np.ndarray, true_bin: Optional[int], true_mit: Optional[int]) -> None:
        pend = self._pending
        self._pending = None
        raw_mse = float(np.mean((pend["pred_state"] - state) ** 2))
        self.last_dynamics_mse = float(min(raw_mse, 1e12)) if np.isfinite(raw_mse) else 1e12
        if true_bin is None or not pend["full_context"]:
            return
        tb = int(true_bin)
        tm = int(true_mit) if true_mit is not None else 0

        if self.use_memory:
            bank: RAMXMemoryBank = self.bank  # type: ignore[assignment]
            bank.ingest(pend["key"], tb, tm, pend["p_att_raw"])

        if self.adapt:
            buf = self._adapt_buf
            buf["seqs"].append(pend["seq"])
            buf["next"].append(state)
            buf["tb"].append(tb)
            buf["tm"].append(tm)
            if len(buf["seqs"]) >= ADAPT_EVERY:
                self._run_ttt()


class StreamingTimesFMHybrid:
    """TimesFM-3 dynamics/forecast + ARY.01's frozen classifier, with an
    optional classify-blend RAM layer on the classifier (`ram_mode`):

      "none"     -- timesfm_ary_cls:    no memory blend at all
      "ungated"  -- fmary_ram_ungated:  classify-blend memory always
                     reads+writes every step (same mechanics as ARY V8)
      "gated"    -- fmary_gated_ram:    same memory, but read+write only
                     when the raw (pre-blend) P(attack) >= gate_threshold

    The classifier reuses a frozen (`ary_base`-mode) `StreamingARY` purely
    for its p_att/p_mit/hidden outputs -- this hybrid's own MemoryBank is
    independent of any standalone `StreamingARY("ary_v8")` instance also
    running in the same round.
    """

    def __init__(
        self,
        ram_mode: str,
        base_model: Optional[nn.Module] = None,
        forecaster: Any = None,
        raw_thresh: Optional[float] = None,
        gate_threshold: float = 0.15,
        tfm_context: int = 100,
        tfm_features: Optional[list[int]] = None,
    ):
        if ram_mode not in RAM_MODES:
            raise ValueError(f"Unknown StreamingTimesFMHybrid ram_mode: {ram_mode}")
        self.ram_mode = ram_mode
        self.gate_threshold = gate_threshold
        self.tfm_context = tfm_context
        self.tfm_features = tfm_features
        self.forecaster = forecaster

        base = base_model if base_model is not None else load_model()
        self._cls = StreamingARY("ary_base", base_model=base)

        self.bank = MemoryBank()
        self.match_thresh = raw_thresh or 30.0
        self._pending: Optional[dict] = None
        self.buffer: list[np.ndarray] = []
        self.last_dynamics_mse: Optional[float] = None
        self.n_steps = 0

    def step(
        self,
        state: np.ndarray,
        true_bin: Optional[int] = None,
        true_mit: Optional[int] = None,
    ) -> dict:
        state = np.asarray(state, dtype=np.float32)
        self.n_steps += 1
        self.buffer.append(state)

        if self._pending is not None:
            self._finalize_pending(true_bin, true_mit)

        cls_out = self._cls.step(state, true_bin=true_bin, true_mit=true_mit)
        p_att_raw, p_mit_raw, hidden = cls_out["p_att"], cls_out["p_mit"], cls_out["hidden"]
        self.last_dynamics_mse = self._cls.last_dynamics_mse

        key = self._cls._padded_seq().reshape(-1)
        p_att, p_mit = p_att_raw, p_mit_raw
        allow_ram = self.ram_mode != "none" and (self.ram_mode == "ungated" or p_att_raw >= self.gate_threshold)
        if allow_ram:
            p_att, p_mit = _classify_blend(self.bank, key, p_att_raw, p_mit_raw, knn_k=1, match_thresh=self.match_thresh)

        full_context = len(self._cls.buffer) >= CONTEXT
        self._pending = {"key": key, "p_att_raw": p_att_raw, "full_context": full_context}

        return {
            "p_att": float(p_att),
            "p_mit": p_mit,
            "p_att_raw": float(p_att_raw),
            "hidden": hidden,
        }

    def _finalize_pending(self, true_bin: Optional[int], true_mit: Optional[int]) -> None:
        pend = self._pending
        self._pending = None
        if self.ram_mode == "none" or true_bin is None or not pend["full_context"]:
            return
        allow_write = self.ram_mode == "ungated" or pend["p_att_raw"] >= self.gate_threshold
        if allow_write:
            self.bank.add(pend["key"], int(true_bin), int(true_mit) if true_mit is not None else 0)

    def forecast(self, horizon: int = 100, feature_indices: Optional[list[int]] = None) -> np.ndarray:
        """`horizon`-step-ahead forecast via TimesFM-3, one univariate series
        per feature. Limited by default to `tfm_features` (a curated subset)
        to keep the live loop responsive -- forecasting all 242 dims every
        window is far too slow for real-time pacing (see plan section 4)."""
        if self.forecaster is None:
            raise RuntimeError("StreamingTimesFMHybrid.forecast() requires a forecaster instance")
        if not self.buffer:
            raise RuntimeError("forecast() called before any step()")

        d = self.buffer[-1].shape[0]
        feats = feature_indices if feature_indices is not None else (self.tfm_features or list(range(d)))
        arr = np.stack(self.buffer[-self.tfm_context :]).astype(np.float32)

        contexts = []
        for j in feats:
            series = arr[:, j]
            if len(series) < self.tfm_context:
                pad_val = series[0] if len(series) else 0.0
                pad = np.full(self.tfm_context - len(series), pad_val, dtype=np.float32)
                series = np.concatenate([pad, series])
            contexts.append(series)

        outs = self.forecaster.predict_batch(
            contexts, horizon=horizon, return_quantiles=False, use_symmetric_averaging=False
        )
        preds = np.zeros((horizon, d), dtype=np.float32)
        for j, o in zip(feats, outs):
            preds[:, j] = np.asarray(o.forecast, dtype=np.float32)
        return preds


class StreamingTimesFMHybridRamx(StreamingTimesFMHybrid):
    """TimesFM dynamics + ARY classifier + RAMX_V.01 classify-blend memory
    (always-on, same as ``fmary_ram_ungated`` but with rolling baseline)."""

    def __init__(
        self,
        base_model: Optional[nn.Module] = None,
        forecaster: Any = None,
        raw_thresh: Optional[float] = None,
        tfm_context: int = 100,
        tfm_features: Optional[list[int]] = None,
        rotation_interval: int = 100,
        dynamic_capacity_pre: int = 20,
        dynamic_capacity_post: int = 40,
        suspicious_thresh: float = 0.01,
    ):
        super().__init__(
            "ungated", base_model=base_model, forecaster=forecaster,
            raw_thresh=raw_thresh, tfm_context=tfm_context, tfm_features=tfm_features,
        )
        self.bank = RAMXMemoryBank(
            rotation_interval=rotation_interval,
            dynamic_capacity_pre=dynamic_capacity_pre,
            dynamic_capacity_post=dynamic_capacity_post,
            suspicious_thresh=suspicious_thresh,
        )

    def _finalize_pending(self, true_bin: Optional[int], true_mit: Optional[int]) -> None:
        pend = self._pending
        self._pending = None
        if true_bin is None or not pend["full_context"]:
            return
        bank: RAMXMemoryBank = self.bank  # type: ignore[assignment]
        bank.ingest(
            pend["key"], int(true_bin),
            int(true_mit) if true_mit is not None else 0,
            pend["p_att_raw"],
        )


def build_all_systems(
    base_model: nn.Module,
    forecaster: Any,
    raw_thresh: float,
    hidden_thresh: float,
    tfm_features: Optional[list[int]] = None,
    gate_threshold: float = 0.15,
) -> dict[str, Any]:
    """Construct all 6 streaming systems compared by the live lab, sharing
    one frozen base model and one TimesFM forecaster instance."""
    return {
        "ary_base": StreamingARY("ary_base", base_model=base_model, raw_thresh=raw_thresh, hidden_thresh=hidden_thresh),
        "ary_v8": StreamingARY("ary_v8", base_model=base_model, raw_thresh=raw_thresh, hidden_thresh=hidden_thresh),
        "ary_v10": StreamingARY("ary_v10", base_model=base_model, raw_thresh=raw_thresh, hidden_thresh=hidden_thresh),
        "timesfm_ary_cls": StreamingTimesFMHybrid(
            "none", base_model=base_model, forecaster=forecaster, raw_thresh=raw_thresh, tfm_features=tfm_features
        ),
        "fmary_ram_ungated": StreamingTimesFMHybrid(
            "ungated", base_model=base_model, forecaster=forecaster, raw_thresh=raw_thresh, tfm_features=tfm_features
        ),
        "fmary_gated_ram": StreamingTimesFMHybrid(
            "gated", base_model=base_model, forecaster=forecaster, raw_thresh=raw_thresh,
            gate_threshold=gate_threshold, tfm_features=tfm_features,
        ),
    }
