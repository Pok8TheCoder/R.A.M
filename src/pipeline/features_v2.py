"""PRISM feature schema v2 — the input representation for YMT.01.

v1 (ZMT.01) describes a single flow with 27 raw numbers. v2 describes the
*window of traffic a flow arrives in*: fan-out, port entropy, protocol mix and
dispersion statistics that are undefined for one flow in isolation.

Two constraints shaped this list:

1. Every column must be computable from both a PCAP and a CIC-IDS-2018 CSV.
   Source IP / destination IP are absent from the CIC CSVs, so IP fan-out
   features are deliberately excluded: they would be zero on every CSV row and
   non-zero on every PCAP row, letting the model recover which corpus a sample
   came from instead of what the traffic is doing.
2. Heavy-tailed rates (bytes/s, IAT, duration) are signed-log compressed before
   aggregation, so a single 10 Gb/s flood does not dominate the window mean.
"""

from __future__ import annotations

# ── Base per-flow record ──────────────────────────────────────────────────────
# Canonical intermediate produced by both the PCAP and CSV readers. It is a
# strict superset of what v1 and v2 each need, so both representations are
# derived from byte-identical flow records and the comparison isolates the
# representation rather than the parser. Column order is load-bearing:
# extract_v2 indexes into it positionally.
BASE_COLS = [
    "dst_port",
    "protocol",
    "duration_us",
    "fwd_pkts",
    "bwd_pkts",
    "fwd_bytes",
    "bwd_bytes",
    "byts_s",
    "pkts_s",
    "iat_mean",
    "iat_std",
    "fwd_len_mean",
    "bwd_len_max",
    "syn_cnt",
    "ack_cnt",
    "fin_cnt",
    "rst_cnt",
    "psh_cnt",
    "payload_mean",
    "ttl_mean",
    "ttl_std",
    "tcpwin_mean",
    "retrans_cnt",
    "frag_cnt",
    # v1-only columns, carried so ZMT.01 can be rebuilt from the same records
    "fwd_len_max",
    "fwd_len_min",
    "bwd_len_min",
    "bwd_len_std",
    "payload_std",
]
BASE_IDX = {c: i for i, c in enumerate(BASE_COLS)}
NUM_BASE = len(BASE_COLS)

# Trailing window length (in flows) used to build one state vector.
# Chosen from data: 232/263 lab captures yield >= 8 flows (median 20).
WINDOW_FLOWS = 8

# ── Block D: quantities aggregated as mean + std across the window ────────────
DISPERSION_BASES = [
    "duration_log",
    "fwd_pkts",
    "bwd_pkts",
    "fwd_bwd_ratio",
    "byts_s_log",
    "pkts_s_log",
    "iat_mean_log",
    "iat_std_log",
    "fwd_len_mean",
    "bwd_len_max",
    "bytes_per_pkt",
    "payload_mean",
    "asymmetry",
]

# ── Full v2 column list ───────────────────────────────────────────────────────
BLOCK_A_COMPOSITION = [
    "win_num_flows",
    "win_uniq_dst_ports",
    "win_dst_port_entropy",
    "win_port_fanout_ratio",
    "win_port_seq_ratio",
    "win_port_range_log",
    "win_proto_entropy",
    "win_flow_rate_log",
]

BLOCK_B_PROTOCOL = [
    "win_proto_frac_tcp",
    "win_proto_frac_udp",
    "win_proto_frac_icmp",
    "win_proto_frac_other",
]

BLOCK_C_PORTCLASS = [
    "win_port_frac_wellknown",
    "win_port_frac_registered",
    "win_port_frac_ephemeral",
    "win_port_frac_web",
    "win_port_frac_remoteadmin",
    "win_port_frac_db",
]

BLOCK_D_DISPERSION = (
    [f"win_mean_{b}" for b in DISPERSION_BASES]
    + [f"win_std_{b}" for b in DISPERSION_BASES]
)

BLOCK_E_FLAGS = [
    "win_syn_frac",
    "win_ack_frac",
    "win_fin_frac",
    "win_rst_frac",
    "win_psh_frac",
    "win_syn_ack_ratio",
    "win_rst_rate",
    "win_flagless_frac",
]

BLOCK_F_PACKET = [
    "win_ttl_mean",
    "win_ttl_std",
    "win_ttl_spread",
    "win_tcpwin_mean_log",
    "win_tcpwin_std_log",
    "win_retrans_rate",
    "win_frag_rate",
    "win_payload_zero_frac",
]

BLOCK_G_CURRENT = [
    "cur_dst_port_log",
    "cur_protocol",
    "cur_duration_log",
    "cur_payload_mean",
]

FEATURE_COLS_V2 = (
    BLOCK_A_COMPOSITION
    + BLOCK_B_PROTOCOL
    + BLOCK_C_PORTCLASS
    + BLOCK_D_DISPERSION
    + BLOCK_E_FLAGS
    + BLOCK_F_PACKET
    + BLOCK_G_CURRENT
)
NUM_FEATURES_V2 = len(FEATURE_COLS_V2)

FEATURE_BLOCKS_V2: dict[str, list[str]] = {
    "A_composition": BLOCK_A_COMPOSITION,
    "B_protocol": BLOCK_B_PROTOCOL,
    "C_portclass": BLOCK_C_PORTCLASS,
    "D_dispersion": BLOCK_D_DISPERSION,
    "E_flags": BLOCK_E_FLAGS,
    "F_packet": BLOCK_F_PACKET,
    "G_current": BLOCK_G_CURRENT,
}

# Port groupings used by block C.
WEB_PORTS = (80, 443, 8080, 8443)
REMOTE_ADMIN_PORTS = (22, 23, 3389, 5900, 5985)
DB_PORTS = (1433, 3306, 5432, 1521, 27017, 6379)
