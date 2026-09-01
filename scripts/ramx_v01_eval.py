#!/usr/bin/env python3
"""RAMX_V.01 — focused eval on CIC-IDS held-out test windows.

Compares:
  V0_frozen        -- no adaptation, no memory
  V10              -- flat hidden-key classify-blend + multitask TTT (k=3)
  V12_ramx_v01     -- V10 backbone + RAMX rolling baseline memory

Also exposes streaming wrappers for incremental (live) use:
  StreamingARYRamxV01  in src/aryan/streaming_variants.py

Usage:
  python scripts/ramx_v01_eval.py
  python scripts/ramx_v01_eval.py --timeline   # include 1000-step kill-chain panel
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
from src.aryan.dataset import load_all_splits  # noqa: E402

OUT = ROOT / "results" / "ram_improve" / "ramx_v01_eval.json"


def calibrate_hidden(base_model, va_s, va_b) -> float:
    keys = []
    for t in range(CONTEXT - 1, len(va_s) - 1):
        seq = va_s[t - CONTEXT + 1 : t + 1]
        keys.append(infer(base_model, seq)["hidden"])
    labels = va_b[CONTEXT : len(va_s)]
    return calibrate_match_thresh(__import__("numpy").stack(keys), labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeline", action="store_true", help="also run 1000-step kill-chain via ram_1000step_eval")
    args = parser.parse_args()

    splits = load_all_splits()
    te_s, te_b, te_m = splits["test"]
    va_s, va_b = splits["val"][0], splits["val"][1]
    base = load_model()
    hidden_thresh = calibrate_hidden(base, va_s, va_b)

    print(f"Hidden match_thresh={hidden_thresh:.2f}\n")
    rows = []

    print("V0 frozen...")
    rows.append(run_variant(
        "V0_frozen", base, te_s, te_b, te_m,
        adapt=False, loss_type="mse", use_memory=False,
        key_space="raw", knn_k=1, match_thresh=None,
    ))

    print("V10 flat RAM...")
    rows.append(run_variant(
        "V10_mt_mem_hidden_k3", base, te_s, te_b, te_m,
        adapt=True, loss_type="multitask", use_memory=True,
        key_space="hidden", knn_k=3, match_thresh=hidden_thresh,
    ))

    print("RAMX_V.01...")
    rows.append(run_variant_ramx(
        "RAMX_V.01", base, te_s, te_b, te_m,
        match_thresh=hidden_thresh, knn_k=3,
    ))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2))

    base_f1 = rows[0]["binary_f1"]
    print(f"\n{'Variant':<24}{'BinF1':>8}{'Prec':>8}{'Rec':>8}{'FPR':>8}")
    print("-" * 56)
    for r in rows:
        d = r["binary_f1"] - base_f1
        print(f"{r['variant']:<24}{r['binary_f1']:>8.3f}{r['binary_precision']:>8.3f}"
              f"{r['binary_recall']:>8.3f}{r['binary_fpr']:>8.3f}  ({d:+.3f})")
    print(f"\nSaved -> {OUT}")

    if args.timeline:
        print("\nRunning 1000-step timeline (ram_1000step_eval)...")
        from scripts import ram_1000step_eval  # noqa: E402
        ram_1000step_eval.main()


if __name__ == "__main__":
    main()
