# RAMX_V.02 — Context-Gated Warmup Calibrator (PRISM V2 path)

RAMX_V.02 is the **Shaun / PRISM V2** adaptive layer: a local z-score
baseline calibrator fused with the transformer classifier, with an explicit
**context gate** so anomaly fusion never fires on prepended calibration data
(e.g. CIC benign warmup windows before a lab PCAP replay).

Implementation: `src/prediction/ramx_v02.py` (`WarmupBaselineCalibrator`,
`RAMXPredictor` with `context_skip_steps`).

Upstream integration: `PRISM` branch `origin/shaun`
(`src/prediction/ramx.py`, commit after v2 gate).

## Problem it solves

When evaluating on lab PCAPs, the standard protocol prepends ~20 benign CIC
windows as context before the attack capture. RAMX v1 fused the relative
anomaly score during that warmup block. Benign CIC states are **out-of-distribution**
relative to the calibrator's local baseline, so fusion pushed `P(attack)` above
0.5 on warmup steps → false positives → F1 stuck at ~0.71 despite 100% detection.

RAMX v2 fixes this by:

1. **Learning** the local mean/std baseline from the context buffer (first
   `warmup_steps` raw states, default 15).
2. **Suppressing fusion** for the first `context_skip_steps` stream steps
   (default 0 in standalone use; set to warmup buffer length in lab eval,
   e.g. 20).
3. **Applying fusion** only on the live capture stream (`step > context_skip`).

This is **not** label leakage — it is scoring-scope hygiene. Warmup windows are
never part of the attack timeline ground truth; they should not contribute alerts.

## Difference from RAMX_V.01

| | RAMX_V.01 | RAMX_V.02 |
|---|---|---|
| Backbone | ARY.01 242-d transformer | PRISM V2 292-d StateTransformer |
| Memory | Hidden-key k-NN classify-blend bank | Warmup z-score + optional episodic bank |
| Domain fix | Rolling quintile baseline + dynamic tier | Local warmup calibrator on raw states |
| Context gate | N/A (ARY uses deferred oracle labels) | **`context_skip_steps`** suppresses fusion on prepended context |
| Typical use | ARY streaming on lab zero-days | Shaun V2 on SchemaAligner-ingested PCAPs |

Both are legitimate RAM variants for different backbones. Do not compare their
headline F1 numbers without noting protocol differences (see caveats below).

## API

```python
from src.prediction.ramx_v02 import RAMXPredictor, RAMX_VERSION

predictor = RAMXPredictor(
    base_model=model,
    scaler_mean=mean,
    scaler_std=std,
    warmup_steps=15,
    context_skip_steps=20,  # CIC context buffer length
    enable_ttt=False,
)

for window in stream:
    res = predictor.predict_state(window_trajectory, device="cpu")
    p_attack = res["p_attack"]
    assert res["ramx_version"] == "2.0"
```

During gated steps, `res["context_gated"] is True` and
`res["p_attack"] == res["raw_p_attack"]`.

## Empirical results (PRISM lab PCAP bench, Sep 2026)

138 attack PCAPs, fair ingest (PCAP → SchemaAligner → StateBuilder @ 15s),
20-step CIC benign warmup prepended:

| Model | Mean F1 | Det rate | Warmup mean P |
|---|---:|---:|---:|
| Shaun V2 + RAMX v1 (no gate) | 0.713 | 100% | 0.185 |
| **Shaun V2 + RAMX v2** | **1.000** | **100%** | **0.156** (base only) |

ARY-5s + RAMX_V.01 on the same PCAP set (5s windows, oracle labels for memory):
F1 ~0.93, det ~99% — strong but not perfect F1 due to occasional benign FPs.

## Honest caveats — is F1=1.0 "cheating"?

**Not cheating in the sense of peeking at attack labels during inference.** RAMX v2
does not use ground-truth labels at runtime. The fix removes incorrect alerts on
steps that are **excluded from attack scoring anyway**.

**But the headline numbers are protocol-specific and optimistic:**

1. **Known attack library** — PCAPs come from the project's labeled lab catalog
   (defacement, cred theft, scans, etc.), not blind field traffic.
2. **Prepended CIC context** — gives the model a benign prior before every replay;
   realistic live deployment would use continuous stream context instead.
3. **Per-PCAP full-attack labels** — F1 assumes the entire lab capture is attack
   traffic after warmup; partial benign tails inside a capture are not modeled.
4. **No adversarial evasion** — spoofability / quiet-lab stress tests are separate;
   ARY base still fails on ultra-quiet lab without RAMX.
5. **ARY RAMX_V.01 uses oracle labels** for memory writes (deferred one step) —
   a stronger assist than Shaun's label-free calibrator. Compare fairly.

**Bottom line:** F1=1.0 on this bench means "perfect separation on our labeled lab
PCAP replays under this warmup protocol" — a real fix for a real scoring bug, not
proof of universal perfect IDS performance.

## Related

- [`RAMX_V01.md`](RAMX_V01.md) — ARY rolling baseline memory
- PRISM `scripts/shaun_eval_sidecar.py` — lab replay driver
- PRISM `results/shaun_vs_ary5/` — comparison reports
