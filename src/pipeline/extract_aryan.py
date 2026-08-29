"""Faithful reimplementation of the `aryan` branch StateBuilder feature schema.

Ports `src/data/state_builder.py::_aggregate_window_vector` from origin/aryan
onto PRISM's base flow records so that schema can be benchmarked head-to-head
against v1 (ZMT.01) and v2 (YMT.01) on identical traffic.

Behaviour is kept as close to the original as possible, including its fallbacks:

* Source/destination IP counts resolve to 0 because neither our base records nor
  the CIC-IDS-2018 CSVs carry IP columns, which is exactly what his
  ``for ... else: features.append(0)`` branches do.
* The URG flag fraction resolves to 0 for the same reason (no URG counter).
* No log compression is applied. The original aggregates raw values.
* Flag fractions divide by flow count, not packet count, matching
  ``total_flag_packets = max(len(window), 1)``.
"""

from __future__ import annotations

import numpy as np

from src.pipeline.features_v2 import BASE_COLS, BASE_IDX

# origin/aryan: src/utils/constants.py
TOP_ATTACKED_PORTS = [21, 22, 23, 25, 53, 80, 110, 135, 139, 143]
TCP_FLAGS = ["SYN", "ACK", "FIN", "RST", "PSH", "URG"]
FLAG_BASE_COL = {
    "SYN": "syn_cnt", "ACK": "ack_cnt", "FIN": "fin_cnt",
    "RST": "rst_cnt", "PSH": "psh_cnt", "URG": None,
}

FEATURE_COLS_ARYAN: list[str] = (
    ["num_flows", "num_unique_src_ips", "num_unique_dst_ips",
     "num_unique_dst_ports", "port_entropy"]
    + [f"{stat}_{c}" for c in BASE_COLS for stat in ("mean", "std")]
    + [f"flag_frac_{f}" for f in TCP_FLAGS]
    + [f"proto_frac_{p}" for p in (6, 17, 1)]
    + [f"top_port_{p}_count" for p in TOP_ATTACKED_PORTS]
)
NUM_FEATURES_ARYAN = len(FEATURE_COLS_ARYAN)


def _uniq_and_entropy(vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w = vals.shape[1]
    c = (vals[:, :, None] == vals[:, None, :]).sum(axis=2).astype(np.float64)
    return (1.0 / c).sum(axis=1), -(np.log2(c / w)).sum(axis=1) / w


def _window_features(win: np.ndarray) -> np.ndarray:
    n, w, _ = win.shape
    ports = win[:, :, BASE_IDX["dst_port"]].astype(np.float64)
    proto = win[:, :, BASE_IDX["protocol"]].astype(np.float64)
    uniq_ports, port_ent = _uniq_and_entropy(ports)

    feats: list[np.ndarray] = [
        np.full(n, float(w)),
        np.zeros(n),          # num_unique_src_ips  — no IP column available
        np.zeros(n),          # num_unique_dst_ips  — no IP column available
        uniq_ports,
        port_ent,
    ]

    vals = win.astype(np.float64)
    for j in range(len(BASE_COLS)):
        feats.append(vals[:, :, j].mean(axis=1))
        feats.append(vals[:, :, j].std(axis=1))

    for flag in TCP_FLAGS:
        col = FLAG_BASE_COL[flag]
        if col is None:
            feats.append(np.zeros(n))
        else:
            feats.append(vals[:, :, BASE_IDX[col]].sum(axis=1) / max(w, 1))

    for pid in (6, 17, 1):
        feats.append((proto == pid).mean(axis=1))

    for port in TOP_ATTACKED_PORTS:
        feats.append((ports == port).sum(axis=1).astype(np.float64))

    out = np.stack(feats, axis=1).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def blocks_to_aryan(blocks: np.ndarray, window: int = 8) -> np.ndarray:
    """Convert (n, T, 29) flow blocks into (n, T, 82) StateBuilder-style states.

    Uses the same trailing-window geometry as v2 so the comparison isolates the
    feature schema rather than the windowing policy.
    """
    n, T, _ = blocks.shape
    out = np.zeros((n, T, NUM_FEATURES_ARYAN), dtype=np.float32)
    for t in range(T):
        lo = max(0, t - window + 1)
        out[:, t, :] = _window_features(blocks[:, lo:t + 1, :])
    return out
