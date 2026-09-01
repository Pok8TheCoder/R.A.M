# RAMX_V.01 — Rolling Benign Baseline Memory

RAMX_V.01 is the current production classify-blend memory layer for streaming
attack detection on the ARY.01 backbone. It replaces the flat, unbounded
`MemoryBank` used in V8/V10 with a **rolling benign baseline** plus a **gated
dynamic tier** for suspicious and confirmed-attack windows.

Implementation: `src/aryan/streaming_variants.py` (`RAMXMemoryBank`,
`StreamingARYRamxV01`).

## Problem it solves

On long benign runs, flat memory accumulates every window forever. Once enough
quiet history piles up, a real attack's k-NN vote gets swamped by irrelevant
"normal" neighbors and never crosses the detection threshold — even when the
raw classifier signal is clearly elevated.

RAMX_V.01 fixes this by:

1. **Rolling baseline (20 slots, 4 quintiles of 5)** — on a regular benign
   stream, new entries land in quintile 4 (slots 16–20). Every
   `rotation_interval` benign steps (default 100), quintiles cascade down
   (Q4→Q3→Q2→Q1) and Q4 clears for fresh writes. Nothing stays frozen
   indefinitely.

2. **Gated dynamic tier (slots 21–60 after first attack)** — starts empty.
   Only windows with raw `P(attack) >= suspicious_thresh` (default 0.01) or
   confirmed attacks (`true_bin == 1`) are stored. Plain benign windows are
   never written here. Capacity grows from 20→40 on first confirmed attack;
   FIFO eviction at capacity.

3. **Same classify-blend read path as V10** — hidden-state keys (256-d),
   k=3 weighted vote, `match_thresh` calibrated on val split. Blend formula
   unchanged from V8/V10 (`BLEND_FLOOR=0.15`, `BLEND_MAX_WEIGHT=0.6`).

## Requirements for RAMX to help

RAMX is a **recall rescue** layer, not a universal booster:

| Requirement | Why |
|---|---|
| Raw `P(attack)` below threshold on OOD | Blend pulls score *up* toward attack-labeled neighbors |
| **Hidden-state** memory keys | Raw 4840-d window keys fail on live/OOD traffic (L2 distances ~500k vs thresh ~859) |
| V10 backbone (multitask TTT + k=3) | Frozen base + raw keys does not detect on lab zero-days |

RAMX does **not** fix upstream miscalibration (e.g. SHNV stuck at ~77%
`P(attack)` on adapted features) or FM hybrids that use frozen `ary_base` +
raw-key RAM.

## API

### Batch (offline replay)

```python
from scripts.ram_improve_eval import run_variant_ramx, load_model

base = load_model()
metrics = run_variant_ramx(
    "RAMX_V.01", base, states, bin_labels, mit_labels,
    match_thresh=hidden_thresh, knn_k=3,
)
```

### Streaming (live / step-by-step)

```python
from src.aryan.streaming_variants import StreamingARYRamxV01, calibrate_thresholds, load_model

base = load_model()
raw_t, hidden_t = calibrate_thresholds(base, va_states, va_bins)
sys = StreamingARYRamxV01(base_model=base, hidden_thresh=hidden_t)

out = sys.step(state_242d, true_bin=prev_label, true_mit=prev_mit)
p_attack = out["p_att"]
```

Labels are deferred by one step (same semantics as batch `run_variant`).

## Eval scripts

| Script | What it runs |
|---|---|
| `scripts/ramx_v01_eval.py` | V0 vs V10 vs RAMX on CIC test split |
| `scripts/ram_improve_eval.py` | Full V0–V12 sweep (includes `V12_ramx_v01`) |
| `scripts/ram_1000step_eval.py` | 1000-step kill-chain timeline + RAMX panel |

## Empirical results (PRISM live lab, Sep 2026)

On captured Docker lab HTTP zero-days (defacement, key theft, SQLi cred theft):

| System | Lab zero-day mean F1 | Notes |
|---|---:|---|
| `ary_base` | 0.000 | Raw P(attack) ~0.001 on OOD |
| `ary_ramx_v01` | **1.000** | Hidden-key neighbor pulls score above 0.5, TTD=0 |
| `shnv_01_base` | 0.286 | Always-on ~67% alarm via 242→110 adapter |
| `shnv_01_ramx` | 0.286 | RAMX cannot fix miscalibrated base |

On 100+ benign windows before attack, flat V10 dilutes and misses; RAMX
maintains elevated `P(attack)` on the attack window where flat memory fails.

## Related classes

- `MemoryBank` — original flat episodic store (V8)
- `TieredMemoryBank` — frozen baseline + gated dynamic (precursor to RAMX)
- `StreamingTimesFMHybridRamx` — TimesFM forecast + RAMX on classifier
  (still uses raw keys on FM path; see PRISM docs for limitations)
