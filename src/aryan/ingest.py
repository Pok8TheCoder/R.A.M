"""Build 242-dim Aryan-WM state sequences from CIC CSV, PCAP, or lab captures.

ARY.01/02 were trained on CIC 30s windows aggregated by StateBuilder. This
module follows that recipe: counts + mean/std of numeric columns + flag/proto
/port extras, then pad/truncate to TARGET_DIM=242 so the trained checkpoint
can run. Missing columns are 0, matching Aryan's own fallbacks.

Live/pcap 242-d is the same builder, not the same distribution as CIC-IDS-2018.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.aryan.constants import CICIDS_LABEL_TO_MITRE, MITRE_STAGES

TARGET_DIM = 242
WINDOW_SEC = 30.0
TOP_PORTS = [21, 22, 23, 25, 53, 80, 110, 135, 139, 143]
FLAG_COLS = [
    "SYN Flag Cnt", "ACK Flag Cnt", "FIN Flag Cnt",
    "RST Flag Cnt", "PSH Flag Cnt", "URG Flag Cnt",
]
EXCLUDE = {
    "window_id", "Label", "label", "mitre_stage", "mitre_stage_id",
    "Timestamp", "timestamp", "Src IP", "src_ip", "Dst IP", "dst_ip",
    "src_ip_hash", "dst_ip_hash", "Flow ID", "flow_id",
}


def _stage_id_from_label(raw) -> int:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return 0
    text = str(raw).strip()
    if text in MITRE_STAGES:
        return MITRE_STAGES[text]
    mapped = CICIDS_LABEL_TO_MITRE.get(text, "Benign")
    return MITRE_STAGES.get(mapped, 0)


def _shannon_entropy(vals: np.ndarray) -> float:
    if len(vals) == 0:
        return 0.0
    _, counts = np.unique(vals, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p + 1e-12)).sum())


def _aggregate_window(window: pd.DataFrame) -> np.ndarray:
    features: list[float] = [float(len(window))]
    for col in ("Src IP", "src_ip", "src"):
        if col in window.columns:
            features.append(float(window[col].nunique()))
            break
    else:
        features.append(0.0)
    for col in ("Dst IP", "dst_ip", "dst"):
        if col in window.columns:
            features.append(float(window[col].nunique()))
            break
    else:
        features.append(0.0)
    for col in ("Dst Port", "dst_port"):
        if col in window.columns:
            ports = window[col].values
            features.append(float(pd.Series(ports).nunique()))
            features.append(_shannon_entropy(ports))
            break
    else:
        features.extend([0.0, 0.0])

    numeric_cols = [
        c for c in window.select_dtypes(include=[np.number]).columns
        if c not in EXCLUDE
    ]
    for col in numeric_cols:
        vals = pd.to_numeric(window[col], errors="coerce").to_numpy(dtype=np.float64)
        features.append(float(np.nanmean(vals)) if len(vals) else 0.0)
        features.append(float(np.nanstd(vals)) if len(vals) else 0.0)

    n = max(len(window), 1)
    for col in FLAG_COLS:
        if col in window.columns:
            features.append(float(pd.to_numeric(window[col], errors="coerce").sum()) / n)
        elif col.split()[0].lower() + "_cnt" in window.columns:
            alt = col.split()[0].lower() + "_cnt"
            features.append(float(pd.to_numeric(window[alt], errors="coerce").sum()) / n)
        else:
            features.append(0.0)

    proto_col = next((c for c in ("Protocol", "protocol", "proto") if c in window.columns), None)
    if proto_col:
        proto = window[proto_col]
        if proto.dtype == object:
            mapped = proto.map({"TCP": 6, "UDP": 17, "ICMP": 1}).fillna(-1)
        else:
            mapped = pd.to_numeric(proto, errors="coerce").fillna(-1)
        for pid in (6, 17, 1):
            features.append(float((mapped == pid).mean()))
    else:
        features.extend([0.0, 0.0, 0.0])

    port_col = next((c for c in ("Dst Port", "dst_port") if c in window.columns), None)
    if port_col:
        ports = pd.to_numeric(window[port_col], errors="coerce").to_numpy()
        for port in TOP_PORTS:
            features.append(float(np.sum(ports == port)))
    else:
        features.extend([0.0] * len(TOP_PORTS))

    vec = np.nan_to_num(np.asarray(features, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if vec.size < TARGET_DIM:
        vec = np.pad(vec, (0, TARGET_DIM - vec.size))
    return vec[:TARGET_DIM]


def _assign_windows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts_col = next((c for c in ("Timestamp", "timestamp", "time") if c in df.columns), None)
    if ts_col is not None:
        ts = pd.to_datetime(df[ts_col], errors="coerce")
        if ts.notna().any():
            t0 = ts.dropna().iloc[0]
            df["window_id"] = ((ts - t0).dt.total_seconds().fillna(0) // WINDOW_SEC).astype(int)
            return df
        numeric = pd.to_numeric(df[ts_col], errors="coerce")
        if numeric.notna().any():
            t0 = float(numeric.dropna().iloc[0])
            df["window_id"] = ((numeric.fillna(t0) - t0) // WINDOW_SEC).astype(int)
            return df
    df["window_id"] = np.arange(len(df)) // 32
    return df


def _states_from_flow_df(df: pd.DataFrame) -> tuple[np.ndarray, list[str], list[tuple[str, int, int]]]:
    if df is None or len(df) == 0:
        empty = np.zeros((0, TARGET_DIM), dtype=np.float32)
        return empty, [], []
    df = _assign_windows(df)
    label_col = next((c for c in ("Label", "label", "mitre_stage") if c in df.columns), None)
    states, labels = [], []
    for wid, window in df.groupby("window_id", sort=True):
        states.append(_aggregate_window(window))
        if label_col:
            raw = window[label_col].mode().iloc[0] if len(window) else "Benign"
            text = str(raw).strip()
            if text in CICIDS_LABEL_TO_MITRE or text == "Benign":
                labels.append("Benign" if text == "Benign" else text)
            else:
                stage = _stage_id_from_label(raw)
                name = next(k for k, v in MITRE_STAGES.items() if v == stage)
                labels.append("Benign" if name == "Benign" else name)
        else:
            labels.append("Benign")
    full = np.stack(states).astype(np.float32) if states else np.zeros((0, TARGET_DIM), dtype=np.float32)
    segments: list[tuple[str, int, int]] = []
    if labels:
        start, cur = 0, labels[0]
        for i, lab in enumerate(labels + [None]):
            if lab != cur:
                segments.append((cur, start, i))
                start, cur = i, lab
    return full, labels, segments


def csv_to_states(path: str | Path) -> tuple[np.ndarray, list[str], list[tuple[str, int, int]]]:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return _states_from_flow_df(df)


def pcap_to_states(path: str | Path) -> tuple[np.ndarray, list[str], list[tuple[str, int, int]]]:
    from src.pipeline.extract import pcap_to_rows

    rows = pcap_to_rows(path)
    if not rows:
        return np.zeros((0, TARGET_DIM), dtype=np.float32), [], []
    df = pd.DataFrame(rows)
    return _states_from_flow_df(df)


def path_to_states(path: str | Path) -> tuple[np.ndarray, list[str], list[tuple[str, int, int]]]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".csv"}:
        return csv_to_states(path)
    if suffix in {".pcap", ".pcapng"}:
        return pcap_to_states(path)
    raise ValueError(f"Unsupported traffic file: {path}")


def lab_live_states(max_files: int = 8) -> tuple[np.ndarray, list[str], list[tuple[str, int, int]]]:
    """Concatenate newest lab capture pcaps into one 242-d timeline."""
    from src.adversarial.lab_config import SAVE_DIR
    from src.model.attack_catalog import resolve_pcap_class

    pcaps = sorted(SAVE_DIR.glob("*.pcap"), key=lambda p: p.stat().st_mtime, reverse=True)[:max_files]
    pcaps = list(reversed(pcaps))
    chunks, labels, segments = [], [], []
    pos = 0
    for pcap in pcaps:
        states, _, _ = pcap_to_states(pcap)
        if len(states) == 0:
            continue
        class_id = resolve_pcap_class(pcap.name) or "Benign"
        lab = "Benign" if class_id == "Benign" else class_id
        chunks.append(states)
        labels.extend([lab] * len(states))
        segments.append((lab, pos, pos + len(states)))
        pos += len(states)
    if not chunks:
        return np.zeros((0, TARGET_DIM), dtype=np.float32), [], []
    return np.concatenate(chunks), labels, segments
