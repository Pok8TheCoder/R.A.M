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
src/pipeline/
  features_v2.py, extract_aryan.py     # feature-name lists only (FEATURE_COLS_V2 / FEATURE_COLS_ARYAN), used for plot labels
data/processed/
  forecast_captures.pkl                # precomputed per-flow trajectories (train/val/test split, two feature schemas: "v2" and "amt")
models/checkpoints/
  forecast_v2_temporal*.pth, forecast_amt_temporal_ctx60.pth   # trained GRU backbone weights
results/forecast/
  v5_adaptive_memory/                  # first RAM.01 prototype results (context=20/horizon=20)
  v6_ram01_killchain/                  # RAM.01 vs frozen models on the synthetic kill-chain (context=60/horizon=40)
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
```

Outputs (plots + JSON summaries) are written under `results/forecast/`.
