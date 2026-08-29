"""RAM.01 vs the frozen Temporal-Y and Temporal-A forecasters, on a long
synthetic multi-attack timeline.

RAM.01 = Receding-horizon Adaptive Memory, v.01
  (the algorithm from the previous experiment: GRU forecaster + test-time
  weight adaptation every horizon-chunk + a surprise-triggered episodic
  memory bank that recognizes recurring attack-stage shapes across
  different captures.)

Protocol (matches the user's spec exactly):
  - read CONTEXT=60 real steps
  - predict the next HORIZON=40 steps
  - reveal the true 40, slide the 60-step context forward by 40
  - repeat for ~1000 steps total

Timeline: benign test captures concatenated as background, with 3 different
REAL attack captures (never used in training or memory-seeding) spliced in at
different points -- T1595_active_scan, T1499_http_flood, T1498_network_dos.
Everything used here is from the TEST split; the memory bank is seeded only
from TRAIN captures, exactly as before.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.adaptive_memory_forecaster import EpisodicMemoryBank, OnlineAdaptive  # noqa: E402
from scripts.train_temporal_forecaster import TemporalForecaster  # noqa: E402
from src.pipeline.extract_aryan import FEATURE_COLS_ARYAN  # noqa: E402
from src.pipeline.features_v2 import FEATURE_COLS_V2  # noqa: E402

DATA = ROOT / "data" / "processed" / "forecast_captures.pkl"
CKPT_DIR = ROOT / "models" / "checkpoints"
OUT_DIR = ROOT / "results" / "forecast" / "v6_ram01_killchain"

CONTEXT = 60
HORIZON = 40
ANOMALY_PCTL = 97.0
MEM_RADIUS = 5
MATCH_MAX_DIST_FALLBACK = 30.0
BLEND_MAX_WEIGHT = 0.6
GAP_LEN = 98
PICK_V2 = ["win_dst_port_entropy", "win_syn_frac", "cur_duration_log"]
PICK_AMT = ["port_entropy", "flag_frac_SYN", "mean_duration_us"]

ATTACK_PICKS = {
    "T1595_active_scan": "pcap_0126_r34_a1_T1595_active_scan_none_20260827T184405Z",
    "T1499_http_flood": None,   # filled in from the 120-flow a2 capture at runtime
    "T1498_network_dos": None,  # filled in from the 242-flow capture at runtime
}


def load_model(tag, ctx_suffix, device):
    ck = torch.load(CKPT_DIR / f"forecast_{tag}_temporal{ctx_suffix}.pth", map_location=device, weights_only=False)
    m = TemporalForecaster(ck["dim"], torch.tensor(ck["max_step"], dtype=torch.float32)).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck["scaler_mean"], ck["scaler_scale"]


def build_timeline(captures, key):
    """Concatenate 8 benign test captures as a pool, splice in 3 attack test
    captures at 3 points. Returns (array, labels, segments)."""
    benign = [c for c in captures if c["split"] == "test" and c["cls"] == "Benign"]
    benign_pool = np.concatenate([c[key] for c in benign])

    by_cid = {c["cid"]: c for c in captures}
    scan = by_cid["pcap_0126_r34_a1_T1595_active_scan_none_20260827T184405Z"]
    flood = next(c for c in captures if c["split"] == "test" and c["cls"] == "T1499_http_flood" and len(c[key]) > 110)
    dos = next(c for c in captures if c["split"] == "test" and c["cls"] == "T1498_network_dos")

    gaps = [benign_pool[i * GAP_LEN:(i + 1) * GAP_LEN] for i in range(4)]
    pieces = [
        ("Benign", gaps[0]), ("T1595_active_scan", scan[key]),
        ("Benign", gaps[1]), ("T1499_http_flood", flood[key]),
        ("Benign", gaps[2]), ("T1498_network_dos", dos[key]),
        ("Benign", gaps[3]),
    ]
    arrs, labels, segments = [], [], []
    pos = 0
    for lbl, arr in pieces:
        arrs.append(arr)
        labels.extend([lbl] * len(arr))
        segments.append((lbl, pos, pos + len(arr)))
        pos += len(arr)
    full = np.concatenate(arrs)
    print(f"  timeline length ({key}): {len(full)}  segments: "
          + ", ".join(f"{lbl}[{a}:{b}]" for lbl, a, b in segments))
    return full, labels, segments


def calibrate_val_error(model, mean, scale, captures, key, context):
    errs = []
    for c in captures:
        if c["split"] != "val":
            continue
        traj = c[key]
        if len(traj) < context + 1:
            continue
        s = (traj - mean) / scale
        for t in range(context - 1, len(traj) - 1):
            seq = torch.from_numpy(s[t - context + 1:t + 1].astype(np.float32)).unsqueeze(0)
            with torch.no_grad():
                pred = model(seq).squeeze(0).numpy()
            errs.append(np.mean((pred - s[t + 1]) ** 2))
    errs = np.array(errs)
    return float(np.percentile(errs, ANOMALY_PCTL))


def seed_memory(base_model, mean, scale, captures, context, horizon, anomaly_thresh):
    """Seed from TRAIN captures with partial (< horizon) final chunks allowed,
    since most captures are shorter than context+horizon."""
    bank = EpisodicMemoryBank()
    n_used = 0
    for c in captures:
        if c["split"] != "train":
            continue
        traj = c["v2"]
        n = len(traj)
        if n < context + 1:
            continue
        n_used += 1
        s = (traj - mean) / scale
        t = context - 1
        while t + 1 < n:
            h = min(horizon, n - 1 - t)
            window_buf = [s[t - context + 1 + i] for i in range(context)]
            buf = list(window_buf)
            preds = []
            with torch.no_grad():
                for _ in range(h):
                    x = torch.from_numpy(np.stack(buf).astype(np.float32)).unsqueeze(0)
                    nxt = base_model(x).squeeze(0).numpy()
                    preds.append(nxt)
                    buf = buf[1:] + [nxt]
            preds = np.stack(preds)
            true_future = s[t + 1:t + 1 + h]
            step_err = np.mean((preds - true_future) ** 2, axis=1)
            peak_idx = int(np.argmax(step_err))
            if step_err[peak_idx] > anomaly_thresh:
                center = t + 1 + peak_idx
                lo, hi = center - MEM_RADIUS, center + MEM_RADIUS + 1
                if lo >= 0 and hi <= n:
                    vec = s[lo:hi].reshape(-1)
                    cont_end = min(n, center + 1 + horizon)
                    continuation = s[center + 1:cont_end] - s[center]
                    bank.add(vec, c["cls"], c["cid"], int(center), continuation)
            t += h
    print(f"  seeded from {n_used} train captures (len>={context+1}), memory bank size={len(bank)}")
    return bank


def calibrate_match_thresh(bank):
    if len(bank) < 10:
        return MATCH_MAX_DIST_FALLBACK
    rng = np.random.default_rng(0)
    idx = rng.choice(len(bank), size=min(200, len(bank)), replace=False)
    same_d, diff_d = [], []
    for i in idx:
        vec, lbl = bank.vecs[i], bank.entries[i]["label"]
        others = [j for j in range(len(bank)) if j != i]
        d = np.linalg.norm(np.stack([bank.vecs[j] for j in others]) - vec[None, :], axis=1)
        labels = [bank.entries[j]["label"] for j in others]
        for dist, lb in zip(d, labels):
            (same_d if lb == lbl else diff_d).append(dist)
    if not same_d or not diff_d:
        return MATCH_MAX_DIST_FALLBACK
    thresh = float((np.percentile(same_d, 40) + np.percentile(diff_d, 10)) / 2)
    print(f"  match threshold calibrated: same-class p40={np.percentile(same_d,40):.2f} "
          f"diff-class p10={np.percentile(diff_d,10):.2f} -> thresh={thresh:.2f}")
    return thresh


def run_frozen(model, mean, scale, full, context, horizon):
    n = len(full)
    s = (full - mean) / scale
    t = context - 1
    all_pred, all_true, steps = [], [], []
    while t + horizon < n:
        buf = [s[t - context + 1 + i] for i in range(context)]
        preds = []
        with torch.no_grad():
            for _ in range(horizon):
                x = torch.from_numpy(np.stack(buf).astype(np.float32)).unsqueeze(0)
                nxt = model(x).squeeze(0).numpy()
                preds.append(nxt)
                buf = buf[1:] + [nxt]
        preds = np.stack(preds)
        all_pred.append(preds * scale + mean)
        all_true.append(full[t + 1:t + 1 + horizon])
        steps.append(t + 1)
        t += horizon
    return np.concatenate(all_pred), np.concatenate(all_true), steps


def run_ram01(base_model, mean, scale, full, labels, context, horizon, mem_bank,
              anomaly_thresh, match_thresh, adapt_lr, adapt_steps, pullback):
    n = len(full)
    s = (full - mean) / scale
    online = OnlineAdaptive(base_model, adapt_lr, pullback)
    t = context - 1
    all_pred, all_true, steps = [], [], []
    anomalies, retrievals = [], []
    pending_match = None
    while t + horizon < n:
        window_buf = [s[t - context + 1 + i] for i in range(context)]
        raw_preds = online.rollout(window_buf, horizon)

        preds = raw_preds
        if pending_match is not None:
            entry, dist = pending_match
            w = min(BLEND_MAX_WEIGHT, max(0.0, 1.0 - dist / match_thresh)) * 0.5 + 0.15
            cont = entry["continuation"]
            k = min(len(cont), horizon)
            recalled = s[t][None, :] + cont[:k]
            preds = raw_preds.copy()
            preds[:k] = (1 - w) * raw_preds[:k] + w * recalled
        pending_match = None

        true_future_s = s[t + 1:t + 1 + horizon]
        all_pred.append(preds * scale + mean)
        all_true.append(full[t + 1:t + 1 + horizon])
        steps.append(t + 1)

        step_err = np.mean((raw_preds - true_future_s) ** 2, axis=1)
        peak_idx = int(np.argmax(step_err))
        if step_err[peak_idx] > anomaly_thresh:
            center = t + 1 + peak_idx
            lo, hi = center - MEM_RADIUS, center + MEM_RADIUS + 1
            if lo >= 0 and hi <= n:
                vec = s[lo:hi].reshape(-1)
                true_label = labels[center]
                anomalies.append({"step": int(center), "label": true_label})
                matches = mem_bank.query(vec, k=1)
                if matches:
                    entry, dist = matches[0]
                    is_match = dist < match_thresh
                    retrievals.append({"step": int(center), "true_label": true_label,
                                        "retrieved_label": entry["label"], "distance": dist,
                                        "match": bool(is_match)})
                    if is_match:
                        pending_match = (entry, dist)
                cont_end = min(n, center + 1 + horizon)
                continuation = s[center + 1:cont_end] - s[center]
                mem_bank.add(vec, true_label, "kill_chain_eval", int(center), continuation)

        seqs, targets = [], []
        for k in range(horizon):
            idx = t + k
            if idx - context + 1 < 0:
                continue
            seqs.append(s[idx - context + 1:idx + 1])
            targets.append(s[idx + 1])
        if seqs:
            online.adapt(torch.from_numpy(np.stack(seqs).astype(np.float32)),
                          torch.from_numpy(np.stack(targets).astype(np.float32)), adapt_steps)

        t += horizon
    return np.concatenate(all_pred), np.concatenate(all_true), steps, anomalies, retrievals


def segment_mse(pred, true, steps, segments, horizon, mean, scale):
    """Break down MSE by which labeled segment each predicted step falls in.
    Computed in STANDARDIZED units -- some raw features (e.g. bytes/sec) have
    natural-unit ranges in the tens of billions, which would otherwise
    dominate a plain squared-error average and make it meaningless."""
    pred_s = (pred - mean) / scale
    true_s = (true - mean) / scale
    out = {}
    idx_map = []
    for st in steps:
        idx_map.extend(range(st, st + horizon))
    idx_map = np.array(idx_map[:len(pred)])
    for lbl, a, b in segments:
        mask = (idx_map >= a) & (idx_map < b)
        if mask.sum() == 0:
            continue
        mse = float(np.mean((pred_s[mask] - true_s[mask]) ** 2))
        out.setdefault(lbl, []).append(mse)
    return {lbl: float(np.mean(v)) for lbl, v in out.items()}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    with open(DATA, "rb") as f:
        captures = pickle.load(f)

    print("Loading frozen Temporal-Y (v2, ctx60) and Temporal-A (amt, ctx60)...")
    y_model, y_mean, y_scale = load_model("v2", "_ctx60", device)
    a_model, a_mean, a_scale = load_model("amt", "_ctx60", device)

    print("\nBuilding synthetic ~1000-step multi-attack timeline...")
    full_v2, labels, segments = build_timeline(captures, "v2")
    full_amt, _, _ = build_timeline(captures, "amt")

    print("\nCalibrating RAM.01 anomaly threshold (v2, val set)...")
    anomaly_thresh = calibrate_val_error(y_model, y_mean, y_scale, captures, "v2", CONTEXT)
    print(f"  anomaly threshold = {anomaly_thresh:.4f}")

    print("\nSeeding RAM.01 episodic memory from TRAIN captures (context=60)...")
    mem_bank = seed_memory(y_model, y_mean, y_scale, captures, CONTEXT, HORIZON, anomaly_thresh)
    from collections import Counter
    print(f"  by class: {dict(Counter(e['label'] for e in mem_bank.entries).most_common(10))}")
    match_thresh = calibrate_match_thresh(mem_bank)

    print("\nRunning frozen Temporal-Y across the timeline...")
    y_pred, y_true, y_steps = run_frozen(y_model, y_mean, y_scale, full_v2, CONTEXT, HORIZON)
    print("Running frozen Temporal-A across the timeline...")
    a_pred, a_true, a_steps = run_frozen(a_model, a_mean, a_scale, full_amt, CONTEXT, HORIZON)
    print("Running RAM.01 (adaptive + episodic memory) across the timeline...")
    ram_pred, ram_true, ram_steps, anomalies, retrievals = run_ram01(
        y_model, y_mean, y_scale, full_v2, labels, CONTEXT, HORIZON, mem_bank,
        anomaly_thresh, match_thresh, adapt_lr=3e-4, adapt_steps=3, pullback=5e-3)

    to_s_y = lambda A: (A - y_mean) / y_scale  # noqa: E731
    to_s_a = lambda A: (A - a_mean) / a_scale  # noqa: E731
    y_mse = float(np.mean((to_s_y(y_pred) - to_s_y(y_true)) ** 2))
    a_mse = float(np.mean((to_s_a(a_pred) - to_s_a(a_true)) ** 2))
    ram_mse = float(np.mean((to_s_y(ram_pred) - to_s_y(ram_true)) ** 2))
    y_seg = segment_mse(y_pred, y_true, y_steps, segments, HORIZON, y_mean, y_scale)
    ram_seg = segment_mse(ram_pred, ram_true, ram_steps, segments, HORIZON, y_mean, y_scale)
    a_seg = segment_mse(a_pred, a_true, a_steps, segments, HORIZON, a_mean, a_scale)

    print("\n" + "=" * 78)
    print(f"Overall forecast MSE (STANDARDIZED units, own schema -- NOT directly comparable across schemas):")
    print(f"  Temporal-Y (frozen, v2)   mse={y_mse:.4f}")
    print(f"  RAM.01     (v2)           mse={ram_mse:.4f}")
    print(f"  Temporal-A (frozen, amt)  mse={a_mse:.4f}")
    print("\nPer-segment MSE (relative to each model's OWN overall mean, i.e. normalized):")
    for lbl in ["Benign", "T1595_active_scan", "T1499_http_flood", "T1498_network_dos"]:
        yv = y_seg.get(lbl, float("nan")) / y_mse
        rv = ram_seg.get(lbl, float("nan")) / ram_mse
        av = a_seg.get(lbl, float("nan")) / a_mse
        print(f"  {lbl:22s} Temporal-Y={yv:6.2f}x   RAM.01={rv:6.2f}x   Temporal-A={av:6.2f}x")

    print(f"\nRAM.01 anomaly/retrieval events across the timeline: {len(anomalies)} surprises, "
          f"{len(retrievals)} queries")
    matched = [r for r in retrievals if r["match"]]
    correct = [r for r in matched if r["retrieved_label"] == r["true_label"]]
    print(f"  {len(matched)} confident matches, {len(correct)} correct "
          f"({100*len(correct)/len(matched) if matched else 0:.1f}% top-1 accuracy)")
    print("\n  Event log (step, true label, retrieved label, correct?):")
    for r in retrievals:
        mark = "OK " if r["retrieved_label"] == r["true_label"] else "no "
        print(f"    step={r['step']:5d}  true={r['true_label']:22s}  "
              f"retrieved={r['retrieved_label']:22s}  [{mark}]  dist={r['distance']:.2f}"
              f"{'  <-- used' if r['match'] else '  (below threshold, not used)'}")

    summary = {
        "overall_mse": {"Temporal-Y": y_mse, "RAM.01": ram_mse, "Temporal-A": a_mse},
        "segment_mse": {"Temporal-Y": y_seg, "RAM.01": ram_seg, "Temporal-A": a_seg},
        "segments": segments,
        "retrieval_events": retrievals,
        "n_anomalies": len(anomalies),
        "match_thresh": match_thresh, "anomaly_thresh": anomaly_thresh,
    }
    with open(OUT_DIR / "killchain_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # ---- plot v2: truth vs Temporal-Y (frozen) vs RAM.01, full timeline ----
    fig, axes = plt.subplots(len(PICK_V2), 1, figsize=(15, 2.6 * len(PICK_V2)), sharex=True)
    fig.suptitle("RAM.01 vs frozen Temporal-Y  ·  synthetic multi-attack timeline "
                 f"(context={CONTEXT}, horizon={HORIZON})", fontsize=12, fontweight="bold")
    idx_map_y = np.array([i for st in y_steps for i in range(st, st + HORIZON)])[:len(y_pred)]
    idx_map_r = np.array([i for st in ram_steps for i in range(st, st + HORIZON)])[:len(ram_pred)]
    colors_seg = {"Benign": "#eef6ee", "T1595_active_scan": "#fde2e2",
                  "T1499_http_flood": "#e2ecfd", "T1498_network_dos": "#fdf0d5"}
    for ax, name in zip(axes, PICK_V2):
        j = FEATURE_COLS_V2.index(name)
        for lbl, a, b in segments:
            if lbl != "Benign":
                ax.axvspan(a, b, color=colors_seg[lbl], alpha=0.6, zorder=0)
        ax.plot(range(len(full_v2)), full_v2[:, j], color="black", lw=1.2, label="actual", zorder=3)
        ax.plot(idx_map_y, y_pred[:, j], color="#e67e22", lw=1.0, ls=":", label="Temporal-Y (frozen)", zorder=2)
        ax.plot(idx_map_r, ram_pred[:, j], color="#27ae60", lw=1.2, label="RAM.01", zorder=4)
        for r in retrievals:
            if r["match"]:
                ax.axvline(r["step"], color="blue", alpha=0.4, lw=1, zorder=1)
        ax.set_ylabel(name, fontsize=8.5)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper right", ncol=3)
    handles_txt = "shaded = inserted attack (scan=red, http_flood=blue-ish, dos=orange) | blue vline = RAM.01 memory match used"
    axes[-1].set_xlabel(f"step across ~1000-step synthetic session\n{handles_txt}")
    plt.tight_layout()
    out1 = OUT_DIR / "killchain_v2_ram01_vs_temporalY.png"
    fig.savefig(out1, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved -> {out1}")

    # ---- plot amt: truth vs Temporal-A (frozen) ----
    fig, axes = plt.subplots(len(PICK_AMT), 1, figsize=(15, 2.6 * len(PICK_AMT)), sharex=True)
    fig.suptitle(f"Frozen Temporal-A  ·  same synthetic timeline, AMT schema (context={CONTEXT}, horizon={HORIZON})",
                 fontsize=12, fontweight="bold")
    idx_map_a = np.array([i for st in a_steps for i in range(st, st + HORIZON)])[:len(a_pred)]
    for ax, name in zip(axes, PICK_AMT):
        j = FEATURE_COLS_ARYAN.index(name)
        for lbl, a, b in segments:
            if lbl != "Benign":
                ax.axvspan(a, b, color=colors_seg[lbl], alpha=0.6, zorder=0)
        ax.plot(range(len(full_amt)), full_amt[:, j], color="black", lw=1.2, label="actual", zorder=3)
        ax.plot(idx_map_a, a_pred[:, j], color="#8e44ad", lw=1.1, ls=":", label="Temporal-A (frozen)", zorder=2)
        ax.set_ylabel(name, fontsize=8.5)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("step across ~1000-step synthetic session")
    plt.tight_layout()
    out2 = OUT_DIR / "killchain_amt_temporalA.png"
    fig.savefig(out2, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out2}")
    print(f"Saved summary -> {OUT_DIR / 'killchain_summary.json'}")


if __name__ == "__main__":
    main()
