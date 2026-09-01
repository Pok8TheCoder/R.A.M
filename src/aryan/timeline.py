"""Synthetic kill-chain timeline builder shared by RAM eval scripts."""

from __future__ import annotations

import numpy as np

from src.aryan.constants import KILLCHAIN_ATTACKS

GAP_COUNT = 4


def _longest_run(labels_mitre, stage_id):
    runs, i = [], 0
    while i < len(labels_mitre):
        j = i
        while j < len(labels_mitre) and labels_mitre[j] == labels_mitre[i]:
            j += 1
        if labels_mitre[i] == stage_id:
            runs.append((i, j))
        i = j
    return max(runs, key=lambda r: r[1] - r[0]) if runs else None


def build_timeline(train, val, test, target_len: int):
    """Splice real held-out attack windows into a benign background."""
    tr_s, _, tr_m = train
    va_s, _, va_m = val
    te_s, _, te_m = test

    ia_lo, ia_hi = _longest_run(va_m, 2)
    lm_lo, lm_hi = _longest_run(te_m, 3)
    im_lo, im_hi = _longest_run(tr_m, 6)

    seg_initial_access = va_s[ia_lo:ia_hi]
    seg_lateral_move = te_s[lm_lo:lm_hi]
    seg_impact = tr_s[im_lo:im_hi]

    benign_pool = np.concatenate([tr_s[tr_m == 0], va_s[va_m == 0], te_s[te_m == 0]])
    attack_total = len(seg_initial_access) + len(seg_lateral_move) + len(seg_impact)
    remaining = max(target_len - attack_total, GAP_COUNT * 10)
    gap_len = remaining // GAP_COUNT
    n_needed = gap_len * GAP_COUNT
    reps = int(np.ceil(n_needed / len(benign_pool)))
    benign_tiled = np.tile(benign_pool, (reps, 1))[:n_needed]
    gaps = [benign_tiled[i * gap_len:(i + 1) * gap_len] for i in range(GAP_COUNT)]

    pieces = [
        ("Benign", gaps[0]),
        (KILLCHAIN_ATTACKS[2], seg_initial_access),
        ("Benign", gaps[1]),
        (KILLCHAIN_ATTACKS[3], seg_lateral_move),
        ("Benign", gaps[2]),
        (KILLCHAIN_ATTACKS[6], seg_impact),
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
    return full, labels, segments
