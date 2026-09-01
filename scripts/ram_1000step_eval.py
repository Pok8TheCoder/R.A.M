"""1000-step synthetic kill-chain comparison: frozen ARY.01 vs original RAM
(TTT-MSE only, matches current dashboard RAM-A.01 behavior) vs the two winning
fixes found in scripts/ram_improve_eval.py:

  base        -- frozen ARY.01, no adaptation, no memory
  orig_ram    -- current dashboard RAM: TTT on dynamics MSE only, no memory
                 classify-blend (the ~0-1% no-op from results/forecast/v8_*)
  V8          -- memory-only classify-blend, raw-window keys, top-1 NN, NO TTT
  V10         -- multi-task TTT + memory classify-blend, hidden-state keys, k=3
  RAMX_V.01   -- V10 + rolling benign baseline (20 slots, quintile cascade) +
                gated dynamic tier for suspicious/attack windows only

Timeline: the same synthetic kill-chain splice the dashboard uses
(`forecast_sessions.build_timeline`) at 1000 steps -- real held-out CIC
attacks (SSH-Bruteforce, Infilteration, DoS attacks-Hulk) spliced into a
benign background, so results are visually comparable to the Forecast Player.

Outputs:
  results/ram_improve/1000step_metrics.json
  results/ram_improve/1000step_comparison.png  (bar chart, 5 metrics)
  results/ram_improve/1000step_timeline.png    (P(attack) over time, 4 panels)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.ram_improve_eval import (  # noqa: E402
    CONTEXT,
    calibrate_match_thresh,
    infer,
    load_model,
    run_variant,
    run_variant_ramx,
)
from src.aryan.constants import STAGE_ID_TO_ATTACK  # noqa: E402
from src.aryan.dataset import load_all_splits  # noqa: E402
from src.aryan.timeline import build_timeline  # noqa: E402

OUT_DIR = ROOT / "results" / "ram_improve"
TARGET_LEN = 1000
ATTACK_TO_STAGE = {v: k for k, v in STAGE_ID_TO_ATTACK.items()}


def labels_to_ids(labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    bin_labels = np.array([0 if lab == "Benign" else 1 for lab in labels], dtype=np.int64)
    mit_labels = np.array([ATTACK_TO_STAGE.get(lab, 0) for lab in labels], dtype=np.int64)
    return bin_labels, mit_labels


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    splits = load_all_splits()
    train, val, test = splits["train"], splits["val"], splits["test"]
    tr_s, tr_b, tr_m = train
    va_s, va_b, va_m = val

    full, labels, segments = build_timeline(train, val, test, TARGET_LEN)
    bin_labels, mit_labels = labels_to_ids(labels)
    attack_segments = [(lbl, a, b) for lbl, a, b in segments if lbl != "Benign"]

    print(f"Timeline: {len(full)} steps, attack rate {bin_labels.mean():.1%}, "
          f"segments: {[(l, a, b) for l, a, b in segments]}\n")

    base_model = load_model()

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
    print(f"  raw_thresh={raw_thresh:.2f}  hidden_thresh={hidden_thresh:.2f}\n")

    configs = [
        ("base", dict(adapt=False, loss_type="mse", use_memory=False,
                      key_space="raw", knn_k=1, match_thresh=None)),
        ("orig_ram", dict(adapt=True, loss_type="mse", use_memory=False,
                          key_space="raw", knn_k=1, match_thresh=None)),
        ("V8_mem_raw_k1", dict(adapt=False, loss_type="mse", use_memory=True,
                               key_space="raw", knn_k=1, match_thresh=raw_thresh)),
        ("V10_mt_mem_hidden_k3", dict(adapt=True, loss_type="multitask", use_memory=True,
                                     key_space="hidden", knn_k=3, match_thresh=hidden_thresh)),
    ]

    results = []
    series = {}
    for name, cfg in configs:
        print(f"Running {name} on {TARGET_LEN}-step timeline...")
        r, p_atts, p_mits = run_variant_with_series(name, base_model, full, bin_labels, mit_labels, **cfg)
        results.append(r)
        series[name] = {"p_atts": p_atts, "p_mits": p_mits}
        print(f"  F1={r['binary_f1']:.3f} Prec={r['binary_precision']:.3f} "
              f"Rec={r['binary_recall']:.3f} FPR={r['binary_fpr']:.3f} "
              f"MitreF1={r['mitre_f1_macro']:.3f}  ({r['elapsed_sec']:.1f}s)")

    print(f"Running RAMX_V.01 on {TARGET_LEN}-step timeline...")
    r_ramx, p_atts, p_mits = run_variant_ramx(
        "RAMX_V.01", base_model, full, bin_labels, mit_labels,
        match_thresh=hidden_thresh, knn_k=3, capture_series=True,
    )
    results.append(r_ramx)
    series["RAMX_V.01"] = {"p_atts": p_atts, "p_mits": p_mits}
    print(f"  F1={r_ramx['binary_f1']:.3f} Prec={r_ramx['binary_precision']:.3f} "
          f"Rec={r_ramx['binary_recall']:.3f} FPR={r_ramx['binary_fpr']:.3f} "
          f"MitreF1={r_ramx['mitre_f1_macro']:.3f}  ({r_ramx['elapsed_sec']:.1f}s)")

    (OUT_DIR / "1000step_metrics.json").write_text(json.dumps(results, indent=2))

    tb_full = bin_labels[CONTEXT:len(full)]
    x = np.arange(CONTEXT, len(full))
    plot_bars(results, OUT_DIR / "1000step_comparison.png")
    plot_timeline(x, tb_full, series, attack_segments, OUT_DIR / "1000step_timeline.png")

    print(f"\nSaved:\n  {OUT_DIR / '1000step_metrics.json'}\n"
          f"  {OUT_DIR / '1000step_comparison.png'}\n"
          f"  {OUT_DIR / '1000step_timeline.png'}")


def run_variant_with_series(name, base_model, states, bin_labels, mit_labels, **cfg):
    """Thin wrapper around run_variant that also captures the p_attack /
    p_mitre time series for plotting (run_variant itself only returns
    aggregate metrics)."""
    import copy
    import time as _time

    import torch.nn as nn
    from sklearn.metrics import f1_score, precision_recall_fscore_support

    from scripts.ram_improve_eval import (
        ADAPT_EVERY, ADAPT_LR, ADAPT_STEPS, BLEND_FLOOR, BLEND_MAX_WEIGHT, PULLBACK, MemoryBank,
    )
    from src.aryan.components import MultiTaskLoss

    t0 = _time.time()
    online = copy.deepcopy(base_model)
    base_params = [p.clone().detach() for p in base_model.parameters()]
    opt = torch.optim.SGD(online.parameters(), lr=ADAPT_LR)
    mse = nn.MSELoss()
    mt_loss = MultiTaskLoss(lambda_dynamics=0.5, lambda_infiltration=1.2, lambda_mitre=1.0)

    bank = MemoryBank()
    n = len(states)
    thresh = cfg["match_thresh"] if cfg["match_thresh"] is not None else 30.0
    p_atts, p_mits, tb_list, tm_list, dyn_errs = [], [], [], [], []
    seqs_buf, next_buf, tb_buf, tm_buf = [], [], [], []

    online.eval()
    for t in range(CONTEXT - 1, n - 1):
        seq = states[t - CONTEXT + 1:t + 1]
        online.eval()
        out = infer(online, seq)
        p_att, p_mit, hidden, pred_state = out["p_att"], out["p_mit"], out["hidden"], out["pred_state"]
        true_next = states[t + 1]
        tb, tm = int(bin_labels[t + 1]), int(mit_labels[t + 1])
        key = hidden if cfg["key_space"] == "hidden" else seq.reshape(-1)

        if cfg["use_memory"] and len(bank) > 0:
            matches = bank.query(key, k=cfg["knn_k"])
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
        p_mits.append(p_mit)
        tb_list.append(tb)
        tm_list.append(tm)
        dyn_errs.append(float(np.mean((pred_state - true_next) ** 2)))

        if cfg["use_memory"]:
            bank.add(key, tb, tm)

        if cfg["adapt"]:
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
                    if cfg["loss_type"] == "mse":
                        loss = mse(out_a["pred_state_mean"], next_t)
                    else:
                        ld = mt_loss(out_a["pred_state_mean"], out_a["pred_state_logvar"], next_t,
                                     out_a["pred_binary"], tb_t, out_a["pred_mitre"], tm_t)
                        loss = ld["total"]
                    reg = sum(((p - b) ** 2).sum() for p, b in zip(online.parameters(), base_params))
                    (loss + PULLBACK * reg).backward()
                    opt.step()
                seqs_buf, next_buf, tb_buf, tm_buf = [], [], [], []

    p_atts_arr = np.array(p_atts)
    tb_arr = np.array(tb_list)
    pred_bin = (p_atts_arr > 0.5).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(tb_arr, pred_bin, average="binary", zero_division=0)
    tn = int(((pred_bin == 0) & (tb_arr == 0)).sum())
    fp = int(((pred_bin == 1) & (tb_arr == 0)).sum())
    fpr = fp / max(fp + tn, 1)
    mit_pred = np.array([int(np.argmax(p)) for p in p_mits])
    tm_arr = np.array(tm_list)
    mit_f1 = f1_score(tm_arr, mit_pred, average="macro", zero_division=0)

    metrics = {
        "variant": name,
        "binary_f1": float(f1),
        "binary_precision": float(prec),
        "binary_recall": float(rec),
        "binary_fpr": float(fpr),
        "mitre_f1_macro": float(mit_f1),
        "dynamics_mse": float(np.mean(dyn_errs)),
        "elapsed_sec": round(_time.time() - t0, 1),
        "n_steps": len(p_atts),
    }
    return metrics, p_atts_arr, np.stack(p_mits)


def plot_bars(results, out_path):
    metrics = ["binary_f1", "binary_precision", "binary_recall", "binary_fpr", "mitre_f1_macro"]
    metric_labels = ["Bin F1", "Precision", "Recall", "FPR", "MITRE F1"]
    names = [r["variant"] for r in results]
    colors = ["#7f8c8d", "#8e44ad", "#27ae60", "#e67e22", "#3498db"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 4.5))
    fig.patch.set_facecolor("#0d1117")
    x = np.arange(len(names))
    for ax, m, ml in zip(axes, metrics, metric_labels):
        vals = [r[m] for r in results]
        bars = ax.bar(x, vals, color=colors[:len(names)])
        ax.set_title(ml, color="white", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right", color="white", fontsize=8)
        ax.tick_params(colors="white")
        ax.set_facecolor("#161b22")
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        ax.set_ylim(0, max(1.0, max(vals) * 1.15))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                     ha="center", color="white", fontsize=8)
    fig.suptitle("RAM Variants on 1000-Step Kill-Chain Timeline — ARY.01 backbone",
                 color="white", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_timeline(x, tb_full, series, attack_segments, out_path):
    names = list(series.keys())
    colors = {"base": "#7f8c8d", "orig_ram": "#8e44ad", "V8_mem_raw_k1": "#27ae60",
              "V10_mt_mem_hidden_k3": "#e67e22", "RAMX_V.01": "#3498db"}

    fig, axes = plt.subplots(len(names), 1, figsize=(16, 2.6 * len(names)), sharex=True)
    fig.patch.set_facecolor("#0d1117")
    if len(names) == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        ax.set_facecolor("#161b22")
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        for lbl, a, b in attack_segments:
            ax.axvspan(a, b, color="#f85149", alpha=0.18, lw=0)
            ax.text((a + b) / 2, 1.05, lbl, color="#f85149", fontsize=7,
                     ha="center", va="bottom")
        p_atts = series[name]["p_atts"]
        n = min(len(x), len(p_atts))
        ax.plot(x[:n], p_atts[:n], color=colors.get(name, "#58a6ff"), lw=1.1, label=name)
        ax.axhline(0.5, color="#c9d1d9", lw=0.8, ls="--", alpha=0.5)
        ax.set_ylim(-0.05, 1.15)
        ax.set_ylabel("P(attack)", color="white", fontsize=9)
        ax.set_title(name, color="white", fontsize=10, loc="left")
        ax.tick_params(colors="white")

    axes[-1].set_xlabel("timeline step", color="white")
    fig.suptitle("P(attack) over 1000-step kill-chain — red bands = ground-truth attack",
                 color="white", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
