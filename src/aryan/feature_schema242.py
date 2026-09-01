"""Semantic blocks for ARY.01/02 242-d CIC StateBuilder vectors.

Layout matches ``origin/aryan`` StateBuilder padding to 242:
  5 meta + 109 flow means + 109 flow stds + 6 flag fracs + 3 proto fracs + 10 top ports.

YMT.01 (v2) uses 64 window features on 8-flow blocks — see ``features_v2.py``.
The 82-d AMT reimplementation in ``extract_aryan.py`` mirrors the *named* subset
of this schema but ARY checkpoints were trained on the full 242-d padded vector.
"""

from __future__ import annotations

TOP_ATTACKED_PORTS = [21, 22, 23, 25, 53, 80, 110, 135, 139, 143]
TCP_FLAGS = ["SYN", "ACK", "FIN", "RST", "PSH", "URG"]
NUM_FLOW_STATS = 109  # mean+std pairs fill (242 - 24) / 2

META_COLS = [
    "num_flows",
    "num_unique_src_ips",
    "num_unique_dst_ips",
    "num_unique_dst_ports",
    "port_entropy",
]

# Always zero on CIC CSV (no IP columns) — safe to drop in ablations.
DEAD_META = ["num_unique_src_ips", "num_unique_dst_ips"]

BLOCK_A_META = META_COLS
BLOCK_B_MEAN = [f"mean_flow_{i}" for i in range(NUM_FLOW_STATS)]
BLOCK_B_STD = [f"std_flow_{i}" for i in range(NUM_FLOW_STATS)]
BLOCK_B_DISPERSION = BLOCK_B_MEAN + BLOCK_B_STD
BLOCK_C_FLAGS = [f"flag_frac_{f}" for f in TCP_FLAGS]
BLOCK_D_PROTO = [f"proto_frac_{p}" for p in (6, 17, 1)]
BLOCK_E_TOP_PORTS = [f"top_port_{p}_count" for p in TOP_ATTACKED_PORTS]

FEATURE_COLS_242 = (
    BLOCK_A_META + BLOCK_B_DISPERSION + BLOCK_C_FLAGS + BLOCK_D_PROTO + BLOCK_E_TOP_PORTS
)
assert len(FEATURE_COLS_242) == 242, len(FEATURE_COLS_242)

FEATURE_BLOCKS_242: dict[str, list[str]] = {
    "A_meta": BLOCK_A_META,
    "dead_meta": DEAD_META,
    "B_mean": BLOCK_B_MEAN,
    "B_std": BLOCK_B_STD,
    "B_dispersion": BLOCK_B_DISPERSION,
    "C_flags": BLOCK_C_FLAGS,
    "D_proto": BLOCK_D_PROTO,
    "E_top_ports": BLOCK_E_TOP_PORTS,
}

# YMT.01 block names for cross-reference in ablation reports.
YMT_BLOCK_SUMMARY = {
    "A_composition": "window flow count, port entropy, fan-out (8-flow window)",
    "B_protocol": "TCP/UDP/ICMP/other mix",
    "C_portclass": "well-known / registered / ephemeral / web / admin / db",
    "D_dispersion": "log-compressed mean+std of 13 dispersion bases",
    "E_flags": "SYN/ACK/FIN/RST/PSH ratios + syn_ack + rst_rate",
    "F_packet": "TTL, TCP window, retrans, frag, payload stats",
    "G_current": "current flow passthrough (4 dims)",
}


def column_indices(
    drop_blocks: list[str] | None = None,
    keep_blocks: list[str] | None = None,
    drop_cols: list[str] | None = None,
    extra_drop: list[int] | None = None,
) -> list[int]:
    """Return sorted column indices into the 242-d state vector."""
    name_to_idx = {n: i for i, n in enumerate(FEATURE_COLS_242)}
    if keep_blocks:
        keep: set[str] = set()
        for b in keep_blocks:
            keep.update(FEATURE_BLOCKS_242[b])
        cols = [name_to_idx[c] for c in FEATURE_COLS_242 if c in keep]
    else:
        dropped: set[str] = set(drop_cols or [])
        for b in drop_blocks or []:
            dropped.update(FEATURE_BLOCKS_242[b])
        cols = [i for i, c in enumerate(FEATURE_COLS_242) if c not in dropped]
    if extra_drop:
        ban = set(extra_drop)
        cols = [i for i in cols if i not in ban]
    return cols


def apply_column_mask(states: "np.ndarray", cols: list[int]):
    import numpy as np
    return states[:, cols].astype(np.float32)


def describe_subset(cols: list[int]) -> dict:
    names = [FEATURE_COLS_242[i] for i in cols]
    blocks_present = {
        b: sum(1 for c in names if c in block)
        for b, block in FEATURE_BLOCKS_242.items()
    }
    return {
        "num_features": len(cols),
        "columns": names,
        "blocks_present": {k: v for k, v in blocks_present.items() if v},
    }
