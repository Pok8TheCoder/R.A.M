"""Prototype of the user's proposed architecture: a receding-horizon forecaster
that (1) keeps adapting its own weights at runtime using freshly-revealed
ground truth ("test-time training"), and (2) writes a snapshot to a persistent
episodic memory bank whenever it gets badly surprised, then queries that bank
on future surprises -- across DIFFERENT captures -- to recognize "I've seen
this shape before" and use that to improve both forecasting and classification.

Concretely, mapped onto the user's own description:
  - "current step 55, 20 in context, predict next 20"    -> WINDOW=20 context,
    HORIZON=20 receding-horizon chunks (Model-Predictive-Control style).
  - "find the error rate and fix them"                    -> OnlineAdaptive:
    a handful of SGD steps on the just-revealed (context, true_next) pairs,
    regularized to stay close to the frozen base weights (so it adapts
    without drifting arbitrarily far -- a lightweight test-time-training).
  - "keep the surrounding steps as cache ... move forward"  -> on a surprise
    spike (z-scored rollout error vs the validation error distribution),
    snapshot a window around the peak into EpisodicMemoryBank, tagged with
    the capture's true class.
  - "during the next intrusion detection, refer to the cache ... classify
    better"                                                -> on a later
    surprise (in this OR a different capture), query the bank by nearest
    neighbor; if close enough, that's a free classification signal, and its
    stored continuation is blended into the NEXT chunk's forecast.

Memory is seeded from anomalies found across every TRAIN capture (never the
held-out TEST captures), then evaluated on TEST captures under three modes:
  frozen          -- plain trained GRU, no adaptation, no memory (control)
  adaptive        -- + test-time training every 20-step chunk, no memory
  adaptive_memory -- + episodic memory read/write (the full proposal)
"""

from __future__ import annotations

import copy
import json
import pickle
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_temporal_forecaster import TemporalForecaster  # noqa: E402
from src.pipeline.features_v2 import FEATURE_COLS_V2  # noqa: E402

DATA = ROOT / "data" / "processed" / "forecast_captures.pkl"
CKPT = ROOT / "models" / "checkpoints" / "forecast_v2_temporal_ctx20.pth"
OUT_DIR = ROOT / "results" / "forecast" / "v5_adaptive_memory"

CONTEXT = 20        # "steps already done ... have steps in context"
HORIZON = 20        # "predict next 20 steps"
ADAPT_STEPS = 3
ADAPT_LR = 3e-4
PULLBACK = 5e-3     # weight decay back toward frozen base weights (prevents drift)
ANOMALY_PCTL = 97.0   # surprise threshold = this percentile of typical val one-step error
                       # (percentile, not mean+z*std, because squared-error residuals are
                       # heavy-tailed -- a few extreme val examples would otherwise blow up std)
MEM_RADIUS = 5        # cached window = +/- 5 steps around the surprise peak
MATCH_MAX_DIST = None  # calibrated empirically below
BLEND_MAX_WEIGHT = 0.6  # cap how much we ever trust a memory match over the model
PICK = ["win_dst_port_entropy", "win_syn_frac", "cur_duration_log"]


class EpisodicMemoryBank:
    """Stores (context window, true label, continuation-after-peak) snippets,
    written on surprise and retrieved by nearest-neighbor L2 distance."""

    def __init__(self):
        self.vecs: list[np.ndarray] = []
        self.entries: list[dict] = []

    def add(self, vec: np.ndarray, label: str, cid: str, step: int, continuation: np.ndarray):
        self.vecs.append(vec)
        self.entries.append({"label": label, "cid": cid, "step": step, "continuation": continuation})

    def query(self, vec: np.ndarray, k: int = 1):
        if not self.vecs:
            return []
        M = np.stack(self.vecs)
        d = np.linalg.norm(M - vec[None, :], axis=1)
        idx = np.argsort(d)[:k]
        return [(self.entries[i], float(d[i])) for i in idx]

    def __len__(self):
        return len(self.vecs)


class OnlineAdaptive:
    """Wraps a frozen base model with a per-capture 'online' copy that takes a
    few SGD steps toward freshly revealed ground truth each chunk, pulled back
    toward the base weights so it can't run away (crude test-time training)."""

    def __init__(self, base_model: nn.Module, lr: float, pullback: float):
        self.model = copy.deepcopy(base_model)
        self.model.train()
        self.base_params = [p.clone().detach() for p in base_model.parameters()]
        self.opt = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.pullback = pullback
        self.crit = nn.MSELoss()

    def adapt(self, seqs: torch.Tensor, targets: torch.Tensor, steps: int) -> float:
        loss_val = 0.0
        for _ in range(steps):
            self.opt.zero_grad()
            pred = self.model(seqs)
            loss = self.crit(pred, targets)
            reg = sum(((p - b) ** 2).sum() for p, b in zip(self.model.parameters(), self.base_params))
            (loss + self.pullback * reg).backward()
            self.opt.step()
            loss_val = loss.item()
        return loss_val

    def rollout(self, window_buf: list[np.ndarray], h: int) -> np.ndarray:
        buf = list(window_buf)
        preds = []
        self.model.eval()
        with torch.no_grad():
            for _ in range(h):
                x = torch.from_numpy(np.stack(buf).astype(np.float32)).unsqueeze(0)
                nxt = self.model(x).squeeze(0).numpy()
                preds.append(nxt)
                buf = buf[1:] + [nxt]
        self.model.train()
        return np.stack(preds)


def frozen_rollout(model: nn.Module, window_buf: list[np.ndarray], h: int) -> np.ndarray:
    buf = list(window_buf)
    preds = []
    with torch.no_grad():
        for _ in range(h):
            x = torch.from_numpy(np.stack(buf).astype(np.float32)).unsqueeze(0)
            nxt = model(x).squeeze(0).numpy()
            preds.append(nxt)
            buf = buf[1:] + [nxt]
    return np.stack(preds)


def calibrate_val_error(model, mean, scale, captures) -> float:
    """One-step (non-recursive) error distribution on val set -> anomaly threshold.
    Uses a percentile (not mean+z*std) since squared-error residuals are heavily
    right-skewed; a handful of extreme val examples would otherwise inflate std
    enough that no rollout ever counts as 'surprising'."""
    errs = []
    for c in captures:
        if c["split"] != "val":
            continue
        traj = c["v2"]
        if len(traj) < CONTEXT + 1:
            continue
        s = (traj - mean) / scale
        for t in range(CONTEXT - 1, len(traj) - 1):
            seq = torch.from_numpy(s[t - CONTEXT + 1:t + 1].astype(np.float32)).unsqueeze(0)
            with torch.no_grad():
                pred = model(seq).squeeze(0).numpy()
            errs.append(np.mean((pred - s[t + 1]) ** 2))
    errs = np.array(errs)
    thresh = float(np.percentile(errs, ANOMALY_PCTL))
    print(f"  val one-step MSE: median={np.median(errs):.4f} p{ANOMALY_PCTL:.0f}={thresh:.4f} "
          f"max={errs.max():.4f}")
    return thresh


def run_capture(cap, base_model, mean, scale, mem_bank, mode, anomaly_thresh,
                 match_thresh, write_memory: bool):
    """mode: 'frozen' | 'adaptive' | 'adaptive_memory'."""
    traj = cap["v2"]
    cid, label = cap["cid"], cap["cls"]
    n = len(traj)
    if n < CONTEXT + HORIZON + 1:
        return None
    s = (traj - mean) / scale

    online = OnlineAdaptive(base_model, ADAPT_LR, PULLBACK) if mode != "frozen" else None
    active_pred_fn = (lambda buf, h: online.rollout(buf, h)) if online else (lambda buf, h: frozen_rollout(base_model, buf, h))

    t = CONTEXT - 1
    all_pred, all_true = [], []
    anomalies, retrievals = [], []
    pending_match = None  # (entry, dist) recognized on the previous chunk, to bias THIS chunk

    while t + HORIZON < n:
        window_buf = [s[t - CONTEXT + 1 + i] for i in range(CONTEXT)]
        raw_preds = active_pred_fn(window_buf, HORIZON)

        preds = raw_preds
        if mode == "adaptive_memory" and pending_match is not None:
            entry, dist = pending_match
            w = min(BLEND_MAX_WEIGHT, max(0.0, 1.0 - dist / match_thresh)) * 0.5 + 0.15
            cont = entry["continuation"]
            k = min(len(cont), HORIZON)
            anchor = s[t]  # last known real state when the match was recognized
            recalled = anchor[None, :] + cont[:k]
            preds = raw_preds.copy()
            preds[:k] = (1 - w) * raw_preds[:k] + w * recalled
        pending_match = None

        true_future = s[t + 1:t + 1 + HORIZON]
        all_pred.append(preds)
        all_true.append(true_future)

        step_err = np.mean((raw_preds - true_future) ** 2, axis=1)
        peak_idx = int(np.argmax(step_err))
        if step_err[peak_idx] > anomaly_thresh:
            center = t + 1 + peak_idx
            lo, hi = center - MEM_RADIUS, center + MEM_RADIUS + 1
            if lo >= 0 and hi <= n:
                vec = s[lo:hi].reshape(-1)
                anomalies.append({"step": int(center), "err": float(step_err[peak_idx])})
                if mode == "adaptive_memory":
                    matches = mem_bank.query(vec, k=1)
                    if matches:
                        entry, dist = matches[0]
                        retrievals.append({"step": int(center), "true_label": label,
                                            "retrieved_label": entry["label"], "distance": dist,
                                            "match": bool(dist < match_thresh)})
                        if dist < match_thresh:
                            pending_match = (entry, dist)
                if write_memory:
                    cont_end = min(n, center + 1 + HORIZON)
                    continuation = s[center + 1:cont_end] - s[center]
                    mem_bank.add(vec, label, cid, int(center), continuation)

        if online is not None:
            seqs, targets = [], []
            for k in range(HORIZON):
                idx = t + k
                if idx - CONTEXT + 1 < 0:
                    continue
                seqs.append(s[idx - CONTEXT + 1:idx + 1])
                targets.append(s[idx + 1])
            if seqs:
                online.adapt(torch.from_numpy(np.stack(seqs).astype(np.float32)),
                              torch.from_numpy(np.stack(targets).astype(np.float32)), ADAPT_STEPS)

        t += HORIZON

    all_pred = np.concatenate(all_pred)
    all_true = np.concatenate(all_true)
    mse = float(np.mean((all_pred - all_true) ** 2))
    return {"mse": mse, "n_steps": len(all_pred), "anomalies": anomalies, "retrievals": retrievals,
            "pred": all_pred, "true": all_true, "cid": cid, "label": label}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA, "rb") as f:
        captures = pickle.load(f)

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    base_model = TemporalForecaster(ck["dim"], torch.tensor(ck["max_step"], dtype=torch.float32))
    base_model.load_state_dict(ck["model"])
    base_model.eval()
    mean, scale = ck["scaler_mean"], ck["scaler_scale"]

    print("Calibrating anomaly threshold from validation residuals...")
    anomaly_thresh = calibrate_val_error(base_model, mean, scale, captures)
    print(f"  -> anomaly if step MSE > {anomaly_thresh:.4f} (p{ANOMALY_PCTL:.0f} of val errors)")

    # ---- SEED phase: build memory purely from TRAIN captures ----
    mem_bank = EpisodicMemoryBank()
    train_caps = [c for c in captures if c["split"] == "train"]
    print(f"\nSeeding memory bank from {len(train_caps)} train captures...")
    for c in train_caps:
        run_capture(c, base_model, mean, scale, mem_bank, "adaptive_memory",
                    anomaly_thresh, match_thresh=1e9, write_memory=True)
    print(f"  memory bank size after seeding: {len(mem_bank)} cases")
    from collections import Counter
    label_counts = Counter(e["label"] for e in mem_bank.entries)
    print(f"  by class: {dict(label_counts.most_common(10))}")

    # ---- Calibrate match threshold: same-class vs diff-class distance separation ----
    print("\nCalibrating match distance threshold (same-class vs different-class NN distances)...")
    rng = np.random.default_rng(0)
    if len(mem_bank) < 10:
        print(f"  WARNING: only {len(mem_bank)} cases in memory, threshold calibration will be noisy")
    sample_idx = rng.choice(len(mem_bank), size=min(200, len(mem_bank)), replace=False)
    same_d, diff_d = [], []
    for i in sample_idx:
        vec, lbl = mem_bank.vecs[i], mem_bank.entries[i]["label"]
        others = [j for j in range(len(mem_bank)) if j != i]
        d = np.linalg.norm(np.stack([mem_bank.vecs[j] for j in others]) - vec[None, :], axis=1)
        labels = [mem_bank.entries[j]["label"] for j in others]
        for dist, lb in zip(d, labels):
            (same_d if lb == lbl else diff_d).append(dist)
    same_d, diff_d = np.array(same_d), np.array(diff_d)
    match_thresh = float((np.percentile(same_d, 40) + np.percentile(diff_d, 10)) / 2)
    print(f"  same-class dist:  mean={same_d.mean():.2f} p40={np.percentile(same_d,40):.2f}")
    print(f"  diff-class dist:  mean={diff_d.mean():.2f} p10={np.percentile(diff_d,10):.2f}")
    print(f"  -> using match_thresh = {match_thresh:.2f}")

    # ---- TEST phase: frozen vs adaptive vs adaptive_memory ----
    test_caps = [c for c in captures if c["split"] == "test"]
    rng.shuffle(test_caps)
    chosen, seen_classes = [], set()
    for c in test_caps:
        if len(c["v2"]) >= CONTEXT + HORIZON + 1 and c["cls"] not in seen_classes:
            chosen.append(c)
            seen_classes.add(c["cls"])
        if len(chosen) >= 14:
            break
    print(f"\nRunning test comparison on {len(chosen)} test captures (classes: {sorted(seen_classes)})")

    results = {"frozen": [], "adaptive": [], "adaptive_memory": []}
    detail_for_plot = None
    for c in chosen:
        for mode in ("frozen", "adaptive", "adaptive_memory"):
            r = run_capture(c, base_model, mean, scale, mem_bank, mode, anomaly_thresh,
                             match_thresh, write_memory=(mode == "adaptive_memory"))
            if r is None:
                continue
            results[mode].append(r)
            if mode == "adaptive_memory" and c["cls"] == "T1595_active_scan":
                detail_for_plot = r

    summary = {}
    for mode, rs in results.items():
        mse = float(np.mean([r["mse"] for r in rs]))
        summary[mode] = {"mean_mse": mse, "n_captures": len(rs)}
    print("\n" + "=" * 70)
    print("Overall forecast MSE across held-out test captures (lower=better):")
    for mode, s in summary.items():
        print(f"  {mode:18s} mse={s['mean_mse']:.4f}  (n={s['n_captures']} captures)")

    all_retrievals = [ret for r in results["adaptive_memory"] for ret in r["retrievals"]]
    matched = [r for r in all_retrievals if r["match"]]
    correct = [r for r in matched if r["retrieved_label"] == r["true_label"]]
    print(f"\nMemory retrieval on TEST anomalies: {len(all_retrievals)} surprise events, "
          f"{len(matched)} produced a confident match (dist < {match_thresh:.2f})")
    if matched:
        print(f"  top-1 classification accuracy of matches: {len(correct)}/{len(matched)} "
              f"= {100*len(correct)/len(matched):.1f}%")
        wrong = [r for r in matched if r["retrieved_label"] != r["true_label"]]
        for r in wrong[:5]:
            print(f"    miss: true={r['true_label']:28s} retrieved={r['retrieved_label']:28s} dist={r['distance']:.2f}")

    summary["retrieval"] = {
        "n_surprise_events": len(all_retrievals), "n_matched": len(matched),
        "n_correct": len(correct),
        "accuracy": (len(correct) / len(matched)) if matched else None,
        "match_thresh": match_thresh,
    }
    with open(OUT_DIR / "adaptive_memory_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # ---- plot the running example capture ----
    if detail_for_plot is not None:
        frozen_r = next((r for r in results["frozen"] if r["cid"] == detail_for_plot["cid"]), None)
        fig, axes = plt.subplots(len(PICK), 1, figsize=(12, 2.6 * len(PICK)), sharex=True)
        fig.suptitle(f"Adaptive + episodic memory  ·  {detail_for_plot['label']}  "
                     f"(context={CONTEXT}, horizon={HORIZON})", fontsize=12, fontweight="bold")
        t = np.arange(len(detail_for_plot["true"]))
        for ax, name in zip(axes, PICK):
            j = FEATURE_COLS_V2.index(name)
            ax.plot(t, detail_for_plot["true"][:, j], color="black", lw=1.8, label="actual")
            if frozen_r is not None:
                ax.plot(t, frozen_r["pred"][:, j], color="#e67e22", lw=1.1, ls=":", label="frozen GRU")
            ax.plot(t, detail_for_plot["pred"][:, j], color="#27ae60", lw=1.4, label="adaptive+memory")
            for a in detail_for_plot["anomalies"]:
                if a["step"] < len(t):
                    ax.axvline(a["step"], color="red", alpha=0.25, lw=1)
            ax.set_ylabel(name, fontsize=8.5)
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=8, loc="upper right")
        axes[-1].set_xlabel("step (red lines = detected surprise / memory-write or -query events)")
        plt.tight_layout()
        out_path = OUT_DIR / "adaptive_memory_example.png"
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSaved example plot -> {out_path}")

    print(f"Saved summary -> {OUT_DIR / 'adaptive_memory_summary.json'}")


if __name__ == "__main__":
    main()
