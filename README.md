# RAM.01 — Receding-horizon Adaptive Memory

RAM.01 is a forecasting algorithm for multivariate time series that **keeps
learning at inference time**. Instead of a model that is frozen after
training and just predicts forward blindly, RAM.01:

1. Predicts a fixed horizon into the future from its current context window.
2. Gets the true values for that horizon revealed (the "receding horizon"
   part — like model-predictive control).
3. Takes a few gradient steps toward that freshly-revealed ground truth
   before predicting the next chunk ("test-time training", regularized so it
   can't drift arbitrarily far from its trained weights).
4. Whenever its prediction error spikes ("surprise"), it snapshots the
   surrounding window into a persistent **episodic memory bank**.
5. On a later surprise — in this or a completely different sequence — it
   queries that bank by nearest-neighbor. A close match gives (a) a free
   classification guess ("I've seen this shape before, it was an X") and
   (b) a recalled continuation that gets blended into the next forecast.

This repo was split out of a larger network-intrusion-detection project
([PRISM](https://github.com/Pok8TheCoder/PRISM)) where RAM.01 was first
prototyped and validated against classical forecasters (Random Forest,
XGBoost, plain MLP autoencoder) and a plain recurrent (GRU) forecaster on
network flow telemetry. **Nothing about the core mechanism is
network-specific** — `EpisodicMemoryBank` and `OnlineAdaptive` in
[`scripts/adaptive_memory_forecaster.py`](scripts/adaptive_memory_forecaster.py)
operate on generic standardized state vectors — so the goal of this repo is
to develop RAM.01 further and eventually generalize it to any regime-shifting
sequential domain (markets, sensor telemetry, etc.), not just packet flows.

## Where it came from

The idea, in the original author's words: a model that "behaves like
training" during runtime — read some steps of context, predict the next
chunk, find the error rate and correct, and keep a notepad/cache of
surprising moments to refer back to later, since intrusions (and plenty of
other real-world processes) unfold in recognizable stages (scan → probe →
exploit → exfiltrate) that tend to repeat in shape even when the exact
timing and values differ.

## Repo layout

```
scripts/
  train_temporal_forecaster.py         # TemporalForecaster: GRU + bounded-delta head (the backbone RAM.01 wraps)
  train_temporal_forecaster_ctx20.py   # trains it with context=20 (matches the first RAM.01 prototype)
  train_temporal_forecaster_ctx60.py   # trains it with context=60/horizon=40 (matches the kill-chain benchmark)
  adaptive_memory_forecaster.py        # RAM.01 core: EpisodicMemoryBank + OnlineAdaptive (TTT) + the v5 experiment
  ram01_kill_chain_eval.py             # the v6 benchmark: RAM.01 vs frozen GRU on a synthetic 1000-step multi-attack timeline
  ram_improve_eval.py                  # ARY.01 RAM variant sweep (V0–V12): TTT loss, memory keys, k-NN classify-blend
  ram_1000step_eval.py                 # 1000-step kill-chain P(attack) comparison: base vs orig RAM vs V8 vs V10 vs RAMX
  ramx_v01_eval.py                     # focused V0 / V10 / RAMX_V.01 comparison on CIC test (+ optional 1000-step)
  timesfm_ram_compare.py               # ARY base/V8/V10 vs Google TimesFM-3 (+ RAM-wrapped TimesFM) on feature forecasts
src/aryan/
  world_model.py, components.py, ...   # ARY.01 242-d temporal transformer backbone RAM wraps in the v7+ experiments
  streaming_variants.py                # RAMX_V.01: RAMXMemoryBank, TieredMemoryBank, StreamingARYRamxV01 (+ FM hybrids)
  timeline.py                          # synthetic kill-chain splice builder (shared by eval scripts)
docs/
  RAMX_V01.md                          # RAMX_V.01 design, API, and empirical notes
  RAMX_V02.md                          # RAMX_V.02 context-gated calibrator (PRISM V2 / Shaun path)
src/prediction/
  ramx_v02.py                          # WarmupBaselineCalibrator + context-gated RAMXPredictor (v2.0)
data/aryan_splits/                     # CIC-IDS-2018 train/val/test NPZ windows for ARY-RAM evals
models/checkpoints/
  forecast_v2_temporal*.pth, forecast_amt_temporal_ctx60.pth   # trained GRU backbone weights
  aryan_world_model_best.pt            # frozen ARY.01 checkpoint used by ram_improve_eval / timesfm_ram_compare
results/forecast/
  v5_adaptive_memory/                  # first RAM.01 prototype results (context=20/horizon=20)
  v6_ram01_killchain/                  # RAM.01 vs frozen models on the synthetic kill-chain (context=60/horizon=40)
results/ram_improve/
  eval.json                            # V0–V11 binary-F1 / MITRE-F1 sweep on held-out test windows
  1000step_*.png/json                  # 1000-step P(attack) timeline: base vs orig RAM vs V8 vs V10
  timesfm_compare_*.png/json           # feature-forecast MSE vs Google TimesFM-3 (+ timesfm_ram)
```

## Results so far

### v5 — first prototype (context=20, horizon=20)

Mean forecast MSE across 11 held-out test sequences, one mode at a time
(frozen GRU → + test-time training only → + episodic memory too):

| mode              | mean MSE | vs. frozen |
|-------------------|---------:|-----------:|
| frozen            |   0.3348 |          — |
| adaptive (TTT)     |   0.3347 |     ~0%    |
| adaptive + memory  |   0.2614 |   **-21.9%** |

Test-time training alone barely moved the needle; adding the episodic
memory bank on top produced a real, if modest, improvement. Memory
retrieval on surprise events: 28/28 surprises produced a confident
nearest-neighbor match, and **69% of those matches were the correct class**
— a usable, if sparse, free classification signal.

### v6 — synthetic kill-chain (context=60, horizon=40, ~1000 steps)

A synthetic session was built by concatenating benign background traffic
with 3 real held-out attack captures spliced in (`T1595_active_scan`,
`T1499_http_flood`, `T1498_network_dos`), then run through the
receding-horizon protocol end to end.

Overall forecast MSE (standardized units):

| model                    | overall MSE |
|--------------------------|------------:|
| Temporal-Y (frozen GRU)  |       8.522 |
| **RAM.01**               |   **8.129** (**-4.6%**) |
| Temporal-A (frozen GRU, different feature schema) | 6.519 (not directly comparable — different features) |

Per-segment MSE (RAM.01 vs. frozen, same schema):

| segment              | Temporal-Y | RAM.01 | change |
|----------------------|-----------:|-------:|-------:|
| Benign               |      6.203 |  6.455 |  +4.1% (slightly worse) |
| T1595_active_scan    |      4.220 |  4.179 |  -1.0% |
| T1499_http_flood     |     34.752 | 31.188 | **-10.3%** |
| T1498_network_dos    |      1.087 |  0.993 |  -8.6% |

Memory picked up 19 surprise events across the run, 13 produced a confident
match, 9/13 (69%) were the correct class — consistent with the v5 finding.

### v7 — ARY.01 backbone + RAM V8/V10 (context=20, classify-blend fix)

Wrapped the frozen **ARY.01** temporal transformer (242-d CIC features) with
the RAM episodic memory + test-time training layer. Key finding from the
dashboard bug post-mortem: the original RAM only blended memory into *dynamics*
forecasts, not into the classification heads — so attack detection barely moved.

Fix: blend k-NN memory votes directly into `P(attack)` and `P(mitre)`:

| variant | TTT | memory keys | k | binary F1 (test) |
|---------|-----|-------------|---|-----------------:|
| V0 frozen | — | — | — | 0.696 |
| V1 orig RAM (TTT-MSE only) | MSE | — | — | ~0.697 |
| **V8** (memory-only) | — | raw window | 1 | **0.724** |
| **V10** (best combo) | multi-task | hidden state | 3 | **0.731** |

See `results/ram_improve/eval.json` and `1000step_comparison.png` for the full
1000-step kill-chain timeline (P(attack) panels with ground-truth attack bands).

### v8 — TimesFM-3 baseline (feature forecast, dynamics-blend RAM)

Compared ARY base/V8/V10 against Google's **TimesFM-3** foundation model
(330M params, zero-shot) on the same 1000-step timeline, forecasting single
242-d features with the black ground-truth line overlaid:

| feature | winner (MSE) | notes |
|---------|-------------|-------|
| `[0] num_flows` (highest variance) | **timesfm** (35k vs ARY 92k) | all models flatten spikes; RAM continuation-blend hurts MSE |
| `[3] num_unique_dst_ports` (2nd highest) | **timesfm** (1.46k vs ARY 1.66k) | smoother signal; same ranking |

RAM's *continuation-blend* (the original v5/v6 design) still degrades feature
MSE when applied to TimesFM or ARY dynamics rollouts — the classify-blend fix
(V8/V10) is the one that helps attack detection. See `timesfm_compare_*.png`.

### v9 — RAMX_V.01 (rolling benign baseline + gated dynamic tier)

Follow-up to V10 for **long-run streaming** where flat memory dilutes attack
votes after 100+ benign windows. See [`docs/RAMX_V01.md`](docs/RAMX_V01.md).

| component | behavior |
|---|---|
| Baseline (20 slots) | 4 quintiles × 5; writes start in Q4; cascade every 100 benign steps |
| Dynamic tier | Suspicious (`P(attack)≥0.01`) or attack only; 20→40 slots on first attack |
| Classify-blend | Same hidden-key k=3 path as V10 |

On PRISM live-lab HTTP zero-days (Sep 2026): `ary_ramx_v01` mean F1 **1.000**
vs `ary_base` **0.000**; rescues OOD recall where raw classifier stays ~0.001.
RAMX requires hidden keys + low raw scores — it does not help miscalibrated
models (SHNV adapter) or FM hybrids on raw-window keys.

```bash
python -m scripts.ramx_v01_eval.py
python -m scripts.ramx_v01_eval.py --timeline   # + 1000-step kill-chain plots
```

### v10 — RAMX_V.02 (context-gated warmup calibrator, PRISM V2)

Follow-up for **Shaun's 292-d StateTransformer** on SchemaAligner-ingested lab
PCAPs. Fixes false-positive fusion during prepended CIC warmup context.
See [`docs/RAMX_V02.md`](docs/RAMX_V02.md).

| variant | mean F1 (138 lab PCAPs) | warmup mean P |
|---|---:|---:|
| Shaun V2 + RAMX v1 (no gate) | 0.713 | 0.185 |
| **Shaun V2 + RAMX v2** | **1.000** | **0.156** (base only) |

Module: `src/prediction/ramx_v02.py` (`RAMX_VERSION = "2.0"`). Upstream:
`PRISM` branch `origin/shaun`.

**Honest takeaway:** the gains are real but modest, and concentrated on the
*attack* segments (where the "recognize a recurring shape" mechanism has
something to grab onto) rather than benign background (which doesn't repeat
in the same way). At the current dataset scale (~340 captures, mean ~92
flows each) the ceiling of both the frozen and adaptive versions is limited
more by data than by algorithm — the interesting open question is whether
RAM.01's memory+TTT layer scales the same way (or better) on top of a
stronger backbone (e.g. a Transformer instead of a GRU), and on domains with
much more available history (e.g. markets), where the memory bank could
grow far larger and the "recurring shape" hypothesis has more to work with.

## Where this is going

- Swap the GRU backbone for a small Transformer (in progress on the PRISM
  side, being ported back here) to see whether memory+TTT helps a stronger
  base model by a similar margin.
- Generalize `EpisodicMemoryBank`/`OnlineAdaptive` away from the
  network-flow-specific glue code in `adaptive_memory_forecaster.py` into a
  domain-agnostic library so the same mechanism can run on other
  regime-shifting time series (e.g. financial markets, where "recurring
  shapes" like breakouts, squeezes, and reversals are a very similar bet).
- Grow the episodic memory bank's capacity/retrieval (currently brute-force
  L2 nearest-neighbor over a Python list) if/when it needs to scale past a
  few thousand entries.

## Running it

```bash
pip install -r requirements.txt

# retrain the GRU backbones (optional — checkpoints are already included)
python -m scripts.train_temporal_forecaster_ctx20
python -m scripts.train_temporal_forecaster_ctx60

# the v5 prototype (context=20/horizon=20, 11 test captures)
python -m scripts.adaptive_memory_forecaster

# the v6 kill-chain benchmark (context=60/horizon=40, ~1000-step synthetic session)
python -m scripts.ram01_kill_chain_eval

# v7: ARY.01 RAM variant sweep + 1000-step P(attack) comparison
python -m scripts.ram_improve_eval
python -m scripts.ram_1000step_eval

# v9: RAMX_V.01 rolling baseline memory (V0 / V10 / RAMX on test split)
python -m scripts.ramx_v01_eval.py
python -m scripts.ramx_v01_eval.py --timeline

# v8: ARY vs Google TimesFM-3 feature forecast (default: highest-variance dim)
python -m scripts.timesfm_ram_compare
python -m scripts.timesfm_ram_compare --feature-rank 2   # second-highest variance
python -m scripts.timesfm_ram_compare --feature-idx 3  # explicit feature index
```

Outputs (plots + JSON summaries) are written under `results/forecast/` and
`results/ram_improve/`.
