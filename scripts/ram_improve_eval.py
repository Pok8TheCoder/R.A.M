"""Experiment harness: try several RAM (Receding-horizon Adaptive Memory)
variants wrapped around the frozen ARY.01 checkpoint and measure whether any
beats the current ~0-1% no-op result from v8 (results/forecast/v8_ram_aryan_killchain/).

Unlike the dashboard's rollout-based prospective heads (which compound
autoregressive error over a 20-step horizon), classification metrics here are
scored with REAL revealed context at every step (one-step-ahead, causal --
never sees the label it's predicting) so results are directly comparable to
ARY.01's original training-eval numbers (test_binary_f1=0.696 etc.) and isolate
what RAM's online adaptation + memory actually change, decoupled from rollout
compounding noise.

Variants:
  V0 frozen          -- control, no adaptation, no memory
  V1 ttt_mse         -- current RAM: TTT on dynamics MSE only, no memory
  V2 ttt_multitask   -- TTT on full multi-task loss (dynamics+infiltration+mitre)
  V3 ttt_mt_mem_raw  -- V2 + episodic memory, raw-window keys, top-1 NN classify-blend
  V4 ttt_mt_mem_hid  -- V2 + episodic memory, HIDDEN-STATE keys, top-1 NN
  V5 ttt_mt_mem_hid_knn -- V4 + k=5 weighted-vote instead of top-1

Run: python scripts/ram_improve_eval.py
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.aryan.components import MultiTaskLoss  # noqa: E402
from src.aryan.dataset import load_all_splits  # noqa: E402
from src.aryan.world_model import TemporalTransformerWorldModel  # noqa: E402

CKPT = ROOT / "models" / "checkpoints" / "aryan_world_model_best.pt"
OUT = ROOT / "results" / "ram_improve" / "eval.json"
CONTEXT = 20
ADAPT_EVERY = 20     # chunk cadence for TTT, matches dashboard HORIZON
ADAPT_STEPS = 3
ADAPT_LR = 3e-4
PULLBACK = 5e-3
BLEND_MAX_WEIGHT = 0.6
BLEND_FLOOR = 0.15


def load_model() -> TemporalTransformerWorldModel:
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    d_state = ck["model_state_dict"]["embedding.proj.weight"].shape[1]
    model = TemporalTransformerWorldModel(d_state=d_state, d_model=256, n_layers=4, n_heads=8, lookback=CONTEXT)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model


def calibrate_match_thresh(keys: np.ndarray, labels: np.ndarray) -> float:
    """Same/diff-label NN distance calibration (as forecast_sessions.py), on val."""
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


class MemoryBank:
    def __init__(self):
        self.keys: list[np.ndarray] = []
        self.bin_labels: list[int] = []
        self.mit_labels: list[int] = []

    def add(self, key, bin_label, mit_label):
        self.keys.append(key)
        self.bin_labels.append(int(bin_label))
        self.mit_labels.append(int(mit_label))

    def query(self, key, k=1):
        if not self.keys:
            return []
        M = np.stack(self.keys)
        d = np.linalg.norm(M - key[None, :], axis=1)
        idx = np.argsort(d)[:k]
        return [(self.bin_labels[i], self.mit_labels[i], float(d[i])) for i in idx]

    def __len__(self):
        return len(self.keys)


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


def run_variant(name: str, base_model: nn.Module, states: np.ndarray, bin_labels: np.ndarray,
                 mit_labels: np.ndarray, *, adapt: bool, loss_type: str, use_memory: bool,
                 key_space: str, knn_k: int, match_thresh: float | None,
                 val_keys_for_thresh: np.ndarray | None = None) -> dict:
    t0 = time.time()
    online = copy.deepcopy(base_model)
    base_params = [p.clone().detach() for p in base_model.parameters()]
    opt = torch.optim.SGD(online.parameters(), lr=ADAPT_LR)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()
    mt_loss = MultiTaskLoss(lambda_dynamics=0.5, lambda_infiltration=1.2, lambda_mitre=1.0)

    bank = MemoryBank()
    n = len(states)
    p_atts, tb_list, p_mits, tm_list = [], [], [], []
    dyn_errs = []
    seqs_buf, next_buf, tb_buf, tm_buf = [], [], [], []

    thresh = match_thresh if match_thresh is not None else 30.0

    online.eval()
    for t in range(CONTEXT - 1, n - 1):
        seq = states[t - CONTEXT + 1:t + 1]
        online.eval()
        out = infer(online, seq)
        p_att, p_mit, hidden, pred_state = out["p_att"], out["p_mit"], out["hidden"], out["pred_state"]
        true_next = states[t + 1]
        tb, tm = int(bin_labels[t + 1]), int(mit_labels[t + 1])

        key = hidden if key_space == "hidden" else seq.reshape(-1)

        if use_memory and len(bank) > 0:
            matches = bank.query(key, k=knn_k)
            close = [m for m in matches if m[2] < thresh]
            if close:
                weights = np.array([max(0.0, 1.0 - d / thresh) for _, _, d in close])
                weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(close)) / len(close)
                w_total = min(BLEND_MAX_WEIGHT, float(np.mean(weights))) * 0.7 + BLEND_FLOOR
                vote_bin = float(np.sum([w * b for (b, _, _), w in zip(close, weights)]))
                p_att = (1 - w_total) * p_att + w_total * vote_bin
                mit_onehot = np.zeros(7, dtype=np.float32)
                for (_, m, _), w in zip(close, weights):
                    mit_onehot[m] += w
                p_mit = (1 - w_total) * p_mit + w_total * mit_onehot

        p_atts.append(p_att)
        tb_list.append(tb)
        p_mits.append(p_mit)
        tm_list.append(tm)
        dyn_errs.append(float(np.mean((pred_state - true_next) ** 2)))

        if use_memory:
            bank.add(key, tb, tm)

        if adapt:
            seqs_buf.append(seq)
            next_buf.append(true_next)
            tb_buf.append(tb)
            tm_buf.append(tm)
            if len(seqs_buf) >= ADAPT_EVERY:
                online.train()
                seqs_t = torch.from_numpy(np.stack(seqs_buf).astype(np.float32))
                next_t = torch.from_numpy(np.stack(next_buf).astype(np.float32))
                tb_t = torch.tensor(tb_buf, dtype=torch.long)
                tm_t = torch.tensor(tm_buf, dtype=torch.long)
                for _ in range(ADAPT_STEPS):
                    opt.zero_grad()
                    out_a = online(seqs_t)
                    if loss_type == "mse":
                        loss = mse(out_a["pred_state_mean"], next_t)
                    else:
                        ld = mt_loss(out_a["pred_state_mean"], out_a["pred_state_logvar"], next_t,
                                     out_a["pred_binary"], tb_t, out_a["pred_mitre"], tm_t)
                        loss = ld["total"]
                    reg = sum(((p - b) ** 2).sum() for p, b in zip(online.parameters(), base_params))
                    (loss + PULLBACK * reg).backward()
                    opt.step()
                seqs_buf, next_buf, tb_buf, tm_buf = [], [], [], []

    p_atts = np.array(p_atts)
    tb_arr = np.array(tb_list)
    pred_bin = (p_atts > 0.5).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(tb_arr, pred_bin, average="binary", zero_division=0)
    tn = int(((pred_bin == 0) & (tb_arr == 0)).sum())
    fp = int(((pred_bin == 1) & (tb_arr == 0)).sum())
    fpr = fp / max(fp + tn, 1)
    mit_pred = np.array([int(np.argmax(p)) for p in p_mits])
    tm_arr = np.array(tm_list)
    mit_f1 = f1_score(tm_arr, mit_pred, average="macro", zero_division=0)

    return {
        "variant": name,
        "binary_f1": float(f1),
        "binary_precision": float(prec),
        "binary_recall": float(rec),
        "binary_fpr": float(fpr),
        "mitre_f1_macro": float(mit_f1),
        "dynamics_mse": float(np.mean(dyn_errs)),
        "elapsed_sec": round(time.time() - t0, 1),
        "n_steps": len(p_atts),
    }


def main():
    # ARY.01's forward pass expects RAW (unstandardized) states -- confirmed
    # from forecast_sessions.py, which never scales `full`/`val_states` before
    # calling model(seq); mean/std there are only used for z-scoring anomaly
    # *magnitude*, never as the model's actual input. Standardizing states
    # before the forward pass (as this script originally did) causes a severe
    # train/inference distribution mismatch and collapses frozen performance.
    splits = load_all_splits()
    tr_s, tr_b, tr_m = splits["train"]
    va_s, va_b, va_m = splits["val"]
    te_s, te_b, te_m = splits["test"]

    base_model = load_model()

    print(f"Test set: {len(te_s)} windows, attack rate {te_b.mean():.1%}\n")

    # Calibrate match thresholds on val (raw-window keys and hidden-state keys)
    print("Calibrating memory thresholds on val split...")
    val_raw_keys, val_hidden_keys = [], []
    for t in range(CONTEXT - 1, len(va_s) - 1):
        seq = va_s[t - CONTEXT + 1:t + 1]
        out = infer(base_model, seq)
        val_raw_keys.append(seq.reshape(-1))
        val_hidden_keys.append(out["hidden"])
    val_raw_keys = np.stack(val_raw_keys)
    val_hidden_keys = np.stack(val_hidden_keys)
    val_bin_labels = va_b[CONTEXT:len(va_s)]
    raw_thresh = calibrate_match_thresh(val_raw_keys, val_bin_labels)
    hidden_thresh = calibrate_match_thresh(val_hidden_keys, val_bin_labels)
    print(f"  raw-window match_thresh={raw_thresh:.2f}  hidden-state match_thresh={hidden_thresh:.2f}\n")

    results = []

    print("Running V0 frozen (control)...")
    results.append(run_variant("V0_frozen", base_model, te_s, te_b, te_m,
                                adapt=False, loss_type="mse", use_memory=False,
                                key_space="raw", knn_k=1, match_thresh=None))

    print("Running V1 ttt_mse (current RAM behavior)...")
    results.append(run_variant("V1_ttt_mse", base_model, te_s, te_b, te_m,
                                adapt=True, loss_type="mse", use_memory=False,
                                key_space="raw", knn_k=1, match_thresh=None))

    print("Running V2 ttt_multitask...")
    results.append(run_variant("V2_ttt_multitask", base_model, te_s, te_b, te_m,
                                adapt=True, loss_type="multitask", use_memory=False,
                                key_space="raw", knn_k=1, match_thresh=None))

    print("Running V3 ttt_mt + memory (raw keys, top-1)...")
    results.append(run_variant("V3_mt_mem_raw_k1", base_model, te_s, te_b, te_m,
                                adapt=True, loss_type="multitask", use_memory=True,
                                key_space="raw", knn_k=1, match_thresh=raw_thresh))

    print("Running V4 ttt_mt + memory (hidden keys, top-1)...")
    results.append(run_variant("V4_mt_mem_hidden_k1", base_model, te_s, te_b, te_m,
                                adapt=True, loss_type="multitask", use_memory=True,
                                key_space="hidden", knn_k=1, match_thresh=hidden_thresh))

    print("Running V5 ttt_mt + memory (hidden keys, k=5 vote)...")
    results.append(run_variant("V5_mt_mem_hidden_k5", base_model, te_s, te_b, te_m,
                                adapt=True, loss_type="multitask", use_memory=True,
                                key_space="hidden", knn_k=5, match_thresh=hidden_thresh))

    print("Running V6 memory-only (hidden keys, k=5, NO TTT)...")
    results.append(run_variant("V6_mem_only_hidden_k5", base_model, te_s, te_b, te_m,
                                adapt=False, loss_type="mse", use_memory=True,
                                key_space="hidden", knn_k=5, match_thresh=hidden_thresh))

    # Attribution ablations: is the lift from multi-task TTT, or purely from
    # fixing the classification-head/memory blend link (dashboard bug)?
    print("Running V7 ttt_mse (current RAM) + memory raw k1...")
    results.append(run_variant("V7_ttt_mse_mem_raw_k1", base_model, te_s, te_b, te_m,
                                adapt=True, loss_type="mse", use_memory=True,
                                key_space="raw", knn_k=1, match_thresh=raw_thresh))

    print("Running V8 frozen (NO TTT) + memory raw k1...")
    results.append(run_variant("V8_frozen_mem_raw_k1", base_model, te_s, te_b, te_m,
                                adapt=False, loss_type="mse", use_memory=True,
                                key_space="raw", knn_k=1, match_thresh=raw_thresh))

    print("Running V9 ttt_mse + memory hidden k1...")
    results.append(run_variant("V9_ttt_mse_mem_hidden_k1", base_model, te_s, te_b, te_m,
                                adapt=True, loss_type="mse", use_memory=True,
                                key_space="hidden", knn_k=1, match_thresh=hidden_thresh))

    print("Running V10 ttt_mt + memory hidden k3...")
    results.append(run_variant("V10_mt_mem_hidden_k3", base_model, te_s, te_b, te_m,
                                adapt=True, loss_type="multitask", use_memory=True,
                                key_space="hidden", knn_k=3, match_thresh=hidden_thresh))

    print("Running V11 frozen (NO TTT) + memory hidden k1...")
    results.append(run_variant("V11_frozen_mem_hidden_k1", base_model, te_s, te_b, te_m,
                                adapt=False, loss_type="mse", use_memory=True,
                                key_space="hidden", knn_k=1, match_thresh=hidden_thresh))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))

    hdr = f"{'Variant':<24}{'BinF1':>8}{'Prec':>8}{'Rec':>8}{'FPR':>8}{'MitreF1':>9}{'MSE':>9}{'sec':>7}"
    print("\n" + hdr)
    print("-" * len(hdr))
    base_f1 = results[0]["binary_f1"]
    for r in results:
        delta = r["binary_f1"] - base_f1
        print(f"{r['variant']:<24}{r['binary_f1']:>8.3f}{r['binary_precision']:>8.3f}"
              f"{r['binary_recall']:>8.3f}{r['binary_fpr']:>8.3f}{r['mitre_f1_macro']:>9.3f}"
              f"{r['dynamics_mse']:>9.2f}{r['elapsed_sec']:>7.1f}"
              f"   (F1 {'+' if delta>=0 else ''}{delta:.3f} vs frozen)")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
