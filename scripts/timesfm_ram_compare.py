"""Compare ARY.01 (base / V8 / V10) against Google's TimesFM-3 foundation
model on the same 1000-step synthetic kill-chain timeline, plus a "for fun"
RAM-augmented TimesFM variant (episodic memory blend, no TTT -- TimesFM is a
frozen 330M-param foundation model, so test-time weight adaptation is out of
scope here; only the memory-continuation blend from RAM is applied).

All five variants forecast the SAME single feature (the highest-variance raw
dimension) so they're visually and numerically comparable on one plot, with
the real ground-truth series drawn in black on every panel.

Variants:
  ary_base    -- frozen ARY.01, no TTT, no memory (full 242-d rollout)
  ary_V8      -- ARY.01 + episodic memory continuation-blend (raw keys, top-1), no TTT
  ary_V10     -- ARY.01 + multi-task TTT + memory continuation-blend (hidden keys, k=3)
  timesfm     -- Google TimesFM-3, zero-shot, growing real causal context
  timesfm_ram -- TimesFM-3 + episodic memory continuation-blend (no TTT)

Outputs:
  results/ram_improve/timesfm_compare_metrics.json
  results/ram_improve/timesfm_compare_bars.png
  results/ram_improve/timesfm_compare_timeline.png
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.ram_improve_eval import (  # noqa: E402
    ADAPT_EVERY, ADAPT_LR, ADAPT_STEPS, BLEND_FLOOR, BLEND_MAX_WEIGHT, CONTEXT, PULLBACK,
    calibrate_match_thresh, infer, load_model,
)
from src.aryan.components import MultiTaskLoss  # noqa: E402
from src.aryan.constants import STAGE_ID_TO_ATTACK  # noqa: E402
from src.aryan.dataset import load_all_splits  # noqa: E402
from src.aryan.feature_schema242 import FEATURE_COLS_242  # noqa: E402
from src.aryan.timeline import build_timeline  # noqa: E402

OUT_DIR = ROOT / "results" / "ram_improve"
TARGET_LEN = 1000
HORIZON = 20
ATTACK_TO_STAGE = {v: k for k, v in STAGE_ID_TO_ATTACK.items()}


def labels_to_ids(labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    bin_labels = np.array([0 if lab == "Benign" else 1 for lab in labels], dtype=np.int64)
    mit_labels = np.array([ATTACK_TO_STAGE.get(lab, 0) for lab in labels], dtype=np.int64)
    return bin_labels, mit_labels


def pick_feature_idx(tr_s: np.ndarray, *, feature_idx: int | None, feature_rank: int) -> int:
    if feature_idx is not None:
        return int(feature_idx)
    order = np.argsort(tr_s.var(axis=0))[::-1]
    rank = max(1, feature_rank)
    return int(order[min(rank - 1, len(order) - 1)])


def output_stem(feature_idx: int, feature_rank: int | None) -> str:
    if feature_rank is not None and feature_rank != 1:
        return f"timesfm_compare_feat{feature_idx}"
    return "timesfm_compare"


class ContMemory:
    """Episodic memory storing (key, continuation-delta) pairs, written on
    surprise and retrieved by L2 nearest neighbor -- the ORIGINAL RAM design
    (dynamics-forecast blend), not the classification-blend fix from V8/V10's
    binary-F1 experiments. Used here since TimesFM has no classification head."""

    def __init__(self):
        self.keys: list[np.ndarray] = []
        self.continuations: list[np.ndarray] = []

    def add(self, key, continuation):
        self.keys.append(key)
        self.continuations.append(continuation)

    def query(self, key, k=1):
        if not self.keys:
            return []
        M = np.stack(self.keys)
        d = np.linalg.norm(M - key[None, :], axis=1)
        idx = np.argsort(d)[:k]
        return [(self.continuations[i], float(d[i])) for i in idx]

    def __len__(self):
        return len(self.keys)


def ary_state_rollout(base_model, states, bin_labels, mit_labels, *, adapt: bool, loss_type: str,
                       use_memory: bool, key_space: str, knn_k: int, match_thresh: float,
                       anomaly_thresh: float) -> np.ndarray:
    """Chunked 20-step autoregressive rollout across the WHOLE timeline,
    optionally with TTT (dynamics or multi-task loss) and/or episodic memory
    continuation-blend on the dynamics forecast (state-level RAM, matches the
    original v5/v8 design -- distinct from the classification-blend fix)."""
    n, d = states.shape
    online = copy.deepcopy(base_model)
    base_params = [p.clone().detach() for p in base_model.parameters()]
    opt = torch.optim.SGD(online.parameters(), lr=ADAPT_LR)
    mse = nn.MSELoss()
    mt_loss = MultiTaskLoss(lambda_dynamics=0.5, lambda_infiltration=1.2, lambda_mitre=1.0)

    bank = ContMemory()
    full_predicted = np.full((n, d), np.nan, dtype=np.float32)
    pending_match = None

    t = CONTEXT - 1
    while t + HORIZON < n:
        window_buf = [states[t - CONTEXT + 1 + i] for i in range(CONTEXT)]
        online.eval()
        buf = list(window_buf)
        raw_preds = []
        hidden_at_start = None
        with torch.no_grad():
            for step in range(HORIZON):
                x = torch.from_numpy(np.stack(buf).astype(np.float32)).unsqueeze(0)
                out = online(x)
                if step == 0:
                    hidden_at_start = out["hidden"][0].numpy()
                nxt = out["pred_state_mean"].squeeze(0).numpy()
                raw_preds.append(nxt)
                buf = buf[1:] + [nxt]
        raw_preds = np.stack(raw_preds)

        preds = raw_preds
        if use_memory and pending_match is not None:
            cont, dist = pending_match
            w = min(BLEND_MAX_WEIGHT, max(0.0, 1.0 - dist / match_thresh)) * 0.5 + BLEND_FLOOR
            k = min(len(cont), HORIZON)
            recalled = states[t][None, :] + cont[:k]
            preds = raw_preds.copy()
            preds[:k] = (1 - w) * raw_preds[:k] + w * recalled
        pending_match = None

        full_predicted[t + 1:t + 1 + HORIZON] = preds
        true_future = states[t + 1:t + 1 + HORIZON]

        if use_memory:
            step_err = np.mean((raw_preds - true_future) ** 2, axis=1)
            peak_idx = int(np.argmax(step_err))
            if step_err[peak_idx] > anomaly_thresh:
                center = t + 1 + peak_idx
                query_key = hidden_at_start if key_space == "hidden" else states[t - CONTEXT + 1:t + 1].reshape(-1)
                cont_end = min(n, center + 1 + HORIZON)
                continuation = states[center + 1:cont_end] - states[center]
                bank.add(query_key, continuation)
                if len(bank) > 1:
                    matches = bank.query(query_key, k=knn_k)
                    if matches:
                        close = [(c, dd) for c, dd in matches if dd < match_thresh]
                        if close:
                            weights = np.array([max(0.0, 1.0 - dd / match_thresh) for _, dd in close])
                            weights = weights / weights.sum()
                            max_len = max(len(c) for c, _ in close)
                            acc = np.zeros((max_len, d), dtype=np.float32)
                            for (c, _), w2 in zip(close, weights):
                                acc[:len(c)] += w2 * c
                            avg_dist = float(np.mean([dd for _, dd in close]))
                            pending_match = (acc, avg_dist)

        if adapt:
            seqs_t = torch.from_numpy(np.stack([states[i - CONTEXT + 1:i + 1] for i in range(t, t + HORIZON)]).astype(np.float32))
            next_t = torch.from_numpy(true_future.astype(np.float32))
            tb_t = torch.tensor(bin_labels[t + 1:t + 1 + HORIZON], dtype=torch.long)
            tm_t = torch.tensor(mit_labels[t + 1:t + 1 + HORIZON], dtype=torch.long)
            online.train()
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

        t += HORIZON

    return full_predicted


def timesfm_rollout(forecaster, series: np.ndarray, use_memory: bool) -> np.ndarray:
    n = len(series)
    chunk_starts = list(range(CONTEXT - 1, n - 1, HORIZON))
    chunk_starts = [t for t in chunk_starts if t + HORIZON < n]
    contexts = [series[:t + 1].astype(np.float32) for t in chunk_starts]

    outputs = list(forecaster.predict_batch(contexts, horizon=HORIZON,
                                             return_quantiles=False, use_symmetric_averaging=False))
    raw_by_chunk = [np.asarray(o.forecast, dtype=np.float32) for o in outputs]

    full_predicted = np.full(n, np.nan, dtype=np.float32)
    if not use_memory:
        for t, raw in zip(chunk_starts, raw_by_chunk):
            full_predicted[t + 1:t + 1 + HORIZON] = raw
        return full_predicted

    bank = ContMemory()
    errs_seen = []
    pending_match = None
    for t, raw in zip(chunk_starts, raw_by_chunk):
        preds = raw
        if pending_match is not None:
            cont, dist = pending_match
            thresh = max(np.std(series[:t + 1]) * 3.0, 1e-6)
            w = min(BLEND_MAX_WEIGHT, max(0.0, 1.0 - dist / thresh)) * 0.5 + BLEND_FLOOR
            k = min(len(cont), HORIZON)
            recalled = series[t] + cont[:k]
            preds = raw.copy()
            preds[:k] = (1 - w) * raw[:k] + w * recalled
        pending_match = None

        full_predicted[t + 1:t + 1 + HORIZON] = preds
        true_future = series[t + 1:t + 1 + HORIZON]
        step_err = (raw - true_future) ** 2
        errs_seen.extend(step_err.tolist())
        surprise_thresh = np.percentile(errs_seen, 90) if len(errs_seen) > 5 else 1e18
        peak_idx = int(np.argmax(step_err))
        if step_err[peak_idx] > surprise_thresh:
            center = t + 1 + peak_idx
            window_lo = max(0, center - 10)
            key = series[window_lo:center + 1]
            key = np.pad(key, (11 - len(key), 0)) if len(key) < 11 else key[-11:]
            cont_end = min(n, center + 1 + HORIZON)
            continuation = series[center + 1:cont_end] - series[center]
            bank.add(key, continuation)
            if len(bank) > 1:
                match_thresh = np.std(series[:center + 1]) * 3.0
                matches = bank.query(key, k=1)
                if matches:
                    cont2, dist = matches[0]
                    if dist < match_thresh:
                        pending_match = (cont2, dist)

    return full_predicted


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--feature-idx", type=int, default=None, help="242-d feature index to plot/score")
    p.add_argument("--feature-rank", type=int, default=1,
                   help="1=highest variance, 2=second highest, etc. (ignored if --feature-idx set)")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    splits = load_all_splits()
    train, val, test = splits["train"], splits["val"], splits["test"]
    tr_s, tr_b, tr_m = train
    va_s, va_b, va_m = val

    full, labels, segments = build_timeline(train, val, test, TARGET_LEN)
    bin_labels, mit_labels = labels_to_ids(labels)
    attack_segments = [(lbl, a, b) for lbl, a, b in segments if lbl != "Benign"]
    n = len(full)

    feature_idx = pick_feature_idx(tr_s, feature_idx=args.feature_idx, feature_rank=args.feature_rank)
    feat_name = FEATURE_COLS_242[feature_idx] if 0 <= feature_idx < len(FEATURE_COLS_242) else f"dim_{feature_idx}"
    stem = output_stem(feature_idx, args.feature_rank if args.feature_idx is None else None)
    print(f"Timeline: {n} steps, feature_idx={feature_idx} ({feat_name}), "
          f"attack rate {bin_labels.mean():.1%}\n")

    base_model = load_model()

    print("Calibrating ARY memory thresholds + anomaly threshold on val split...")
    val_raw_keys, val_hidden_keys, val_errs = [], [], []
    for t in range(CONTEXT - 1, len(va_s) - 1):
        seq = va_s[t - CONTEXT + 1:t + 1]
        out = infer(base_model, seq)
        val_raw_keys.append(seq.reshape(-1))
        val_hidden_keys.append(out["hidden"])
        val_errs.append(float(np.mean((out["pred_state"] - va_s[t + 1]) ** 2)))
    val_raw_keys = np.stack(val_raw_keys)
    val_hidden_keys = np.stack(val_hidden_keys)
    val_bin_labels = va_b[CONTEXT:len(va_s)]
    raw_thresh = calibrate_match_thresh(val_raw_keys, val_bin_labels)
    hidden_thresh = calibrate_match_thresh(val_hidden_keys, val_bin_labels)
    anomaly_thresh = float(np.percentile(val_errs, 90))
    print(f"  raw_thresh={raw_thresh:.2f}  hidden_thresh={hidden_thresh:.2f}  anomaly_thresh={anomaly_thresh:.2f}\n")

    results = {}
    t0 = time.time()
    print("Running ary_base...")
    results["ary_base"] = ary_state_rollout(base_model, full, bin_labels, mit_labels,
                                             adapt=False, loss_type="mse", use_memory=False,
                                             key_space="raw", knn_k=1, match_thresh=raw_thresh,
                                             anomaly_thresh=anomaly_thresh)
    print(f"  {time.time()-t0:.1f}s")

    t0 = time.time()
    print("Running ary_V8 (memory continuation-blend, raw keys, top-1, no TTT)...")
    results["ary_V8"] = ary_state_rollout(base_model, full, bin_labels, mit_labels,
                                          adapt=False, loss_type="mse", use_memory=True,
                                          key_space="raw", knn_k=1, match_thresh=raw_thresh,
                                          anomaly_thresh=anomaly_thresh)
    print(f"  {time.time()-t0:.1f}s")

    t0 = time.time()
    print("Running ary_V10 (multi-task TTT + memory, hidden keys, k=3)...")
    results["ary_V10"] = ary_state_rollout(base_model, full, bin_labels, mit_labels,
                                           adapt=True, loss_type="multitask", use_memory=True,
                                           key_space="hidden", knn_k=3, match_thresh=hidden_thresh,
                                           anomaly_thresh=anomaly_thresh)
    print(f"  {time.time()-t0:.1f}s")

    print("\nLoading TimesFM-3...")
    t0 = time.time()
    from timesfm3 import ModelConfig, TimesFM3Evaluator
    config = ModelConfig(checkpoint_path="google/timesfm-3.0-pytorch", per_core_batch_size=32, device="cuda")
    forecaster = TimesFM3Evaluator(config)
    print(f"  loaded in {time.time()-t0:.1f}s")

    series = full[:, feature_idx]
    t0 = time.time()
    print("Running timesfm (zero-shot, growing causal context)...")
    tfm_base_1d = timesfm_rollout(forecaster, series, use_memory=False)
    print(f"  {time.time()-t0:.1f}s")

    t0 = time.time()
    print("Running timesfm_ram (memory continuation-blend, no TTT)...")
    tfm_ram_1d = timesfm_rollout(forecaster, series, use_memory=True)
    print(f"  {time.time()-t0:.1f}s")

    # Extract the single plotted feature from ARY's full 242-d rollouts
    series_by_variant = {
        "ary_base": results["ary_base"][:, feature_idx],
        "ary_V8": results["ary_V8"][:, feature_idx],
        "ary_V10": results["ary_V10"][:, feature_idx],
        "timesfm": tfm_base_1d,
        "timesfm_ram": tfm_ram_1d,
    }

    actual = series
    metrics = []
    for name, pred in series_by_variant.items():
        mask = ~np.isnan(pred)
        mse = float(np.mean((pred[mask] - actual[mask]) ** 2)) if mask.sum() else float("nan")
        mae = float(np.mean(np.abs(pred[mask] - actual[mask]))) if mask.sum() else float("nan")
        metrics.append({"variant": name, "mse": mse, "mae": mae, "n_covered": int(mask.sum())})
        print(f"{name:<14} MSE={mse:.4f}  MAE={mae:.4f}  covered={int(mask.sum())}/{n}")

    metrics_path = OUT_DIR / f"{stem}_metrics.json"
    bars_path = OUT_DIR / f"{stem}_bars.png"
    timeline_path = OUT_DIR / f"{stem}_timeline.png"

    metrics_path.write_text(json.dumps({
        "feature_idx": feature_idx,
        "feature_name": feat_name,
        "target_len": n,
        "metrics": metrics,
    }, indent=2))

    plot_bars(metrics, bars_path, feature_idx, feat_name)
    plot_timeline(actual, series_by_variant, attack_segments, feature_idx, feat_name, timeline_path)

    print(f"\nSaved:\n  {metrics_path}\n  {bars_path}\n  {timeline_path}")


def plot_bars(metrics, out_path, feature_idx: int, feat_name: str):
    names = [m["variant"] for m in metrics]
    mses = [m["mse"] for m in metrics]
    colors = ["#7f8c8d", "#27ae60", "#e67e22", "#3498db", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    bars = ax.bar(names, mses, color=colors[:len(names)])
    for b, v in zip(bars, mses):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3g}", ha="center", va="bottom", color="white", fontsize=9)
    ax.set_title(f"Forecast MSE — feature[{feature_idx}] {feat_name}, 1000-step timeline",
                 color="white", fontsize=12)
    ax.tick_params(colors="white")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_timeline(actual, series_by_variant, attack_segments, feature_idx, feat_name, out_path):
    names = list(series_by_variant.keys())
    colors = {"ary_base": "#7f8c8d", "ary_V8": "#27ae60", "ary_V10": "#e67e22",
              "timesfm": "#3498db", "timesfm_ram": "#e74c3c"}
    n = len(actual)
    x = np.arange(n)

    fig, axes = plt.subplots(len(names), 1, figsize=(16, 2.6 * len(names)), sharex=True)
    fig.patch.set_facecolor("#0d1117")

    for ax, name in zip(axes, names):
        ax.set_facecolor("#161b22")
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        for lbl, a, b in attack_segments:
            ax.axvspan(a, b, color="#f85149", alpha=0.15, lw=0)
            ax.text((a + b) / 2, np.nanmax(actual) * 1.05,
                     lbl, color="#f85149", fontsize=7, ha="center", va="bottom")
        ax.plot(x, actual, color="black", lw=1.4, label="actual", zorder=5)
        pred = series_by_variant[name]
        ax.plot(x, pred, color=colors.get(name, "#58a6ff"), lw=1.0, label=name, alpha=0.9, zorder=4)
        ax.set_title(name, color="white", fontsize=10, loc="left")
        ax.tick_params(colors="white")
        ax.legend(loc="upper right", fontsize=7, facecolor="#161b22", labelcolor="white", framealpha=0.6)

    axes[-1].set_xlabel("timeline step", color="white")
    fig.suptitle(f"Feature[{feature_idx}] {feat_name} — black = ground truth, red bands = attack segments",
                 color="white", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
