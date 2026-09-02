"""
PRISM RAMX: Receding-horizon Adaptive Memory with Test-Time Adaptation (TTT)
Solves domain shift and timescale mismatch on raw lab PCAPs and unseen networks.

RAMX v2.0 (context-gated): learn the local baseline during an external context
buffer (e.g. CIC benign warmup), but suppress anomaly fusion until the live
capture stream starts. Prevents false positives on prepended calibration data.

Components:
1. WarmupBaselineCalibrator: Dynamically measures local network noise floor during warmup
2. OnlineAdaptiveTransformer: Performs test-time weight adaptation anchored by L2 pullback
3. EpisodicMemoryBank: Stores surprise attack prototypes and retrieves nearest neighbors
4. RAMXPredictor: Unified production wrapper for high detection rate across arbitrary networks
"""

RAMX_VERSION = "2.0"

import copy
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, List, Optional


class WarmupBaselineCalibrator:
    """
    Calibrates local network baseline statistics during the initial warmup period.
    Eliminates domain shift between large public training datasets and small lab PCAPs.
    """

    def __init__(self, warmup_steps: int = 15, epsilon: float = 1e-4):
        self.warmup_steps = warmup_steps
        self.epsilon = epsilon
        self.warmup_buffer: List[np.ndarray] = []
        self.baseline_mean: Optional[np.ndarray] = None
        self.baseline_std: Optional[np.ndarray] = None
        self.calibrated = False

    def update(self, state_vector: np.ndarray) -> None:
        """Adds a state vector to the warmup buffer."""
        if not self.calibrated:
            self.warmup_buffer.append(state_vector.copy())
            if len(self.warmup_buffer) >= self.warmup_steps:
                buf = np.stack(self.warmup_buffer, axis=0)
                self.baseline_mean = np.mean(buf, axis=0)
                self.baseline_std = np.std(buf, axis=0) + self.epsilon
                self.calibrated = True

    def get_relative_anomaly_score(self, current_state: np.ndarray) -> float:
        """
        Computes z-scored distance of current state relative to the local warmup baseline.
        Returns a normalized anomaly intensity in [0.0, 1.0].
        """
        if not self.calibrated or self.baseline_mean is None or self.baseline_std is None:
            return 0.0

        z_scores = np.abs((current_state - self.baseline_mean) / self.baseline_std)
        # Focus on top 5% most anomalous dimensions (bursts, entropy shifts, degrees)
        top_k = max(5, int(0.05 * len(current_state)))
        top_anomalies = np.sort(z_scores)[-top_k:]
        mean_top_z = float(np.mean(top_anomalies))
        # Sigmoid compression: background Gaussian top-5% sits around ~2.5 (prob ~0.15)
        # Real attacks create z >= 5.0 (prob > 0.60) to z >= 8.0 (prob > 0.94)
        anomaly_prob = float(1.0 / (1.0 + np.exp(-0.8 * (mean_top_z - 4.5))))
        return float(np.clip(anomaly_prob, 0.0, 1.0))



class EpisodicMemoryBank:
    """
    Stores surprise attack signatures and retrieves them on subsequent spikes.
    Enables zero-shot prototype matching across distinct capture streams.
    """

    def __init__(self, max_entries: int = 500):
        self.max_entries = max_entries
        self.vectors: List[np.ndarray] = []
        self.metadata: List[Dict[str, any]] = []

    def add(self, state_trajectory: np.ndarray, label: int, stage: int, info: Optional[Dict] = None) -> None:
        """Stores an episodic snapshot."""
        flat_repr = state_trajectory.flatten()
        if len(self.vectors) >= self.max_entries:
            self.vectors.pop(0)
            self.metadata.pop(0)
        self.vectors.append(flat_repr)
        self.metadata.append({"label": label, "stage": stage, "info": info or {}})

    def query(self, current_trajectory: np.ndarray, k: int = 3) -> List[Tuple[Dict[str, any], float]]:
        """Finds nearest neighbor attack prototypes via Euclidean distance."""
        if not self.vectors:
            return []
        M = np.stack(self.vectors, axis=0)
        target = current_trajectory.flatten()[None, :]
        dists = np.linalg.norm(M - target, axis=1)
        k_nearest = min(k, len(dists))
        best_indices = np.argsort(dists)[:k_nearest]
        return [(self.metadata[idx], float(dists[idx])) for idx in best_indices]

    def __len__(self) -> int:
        return len(self.vectors)


class OnlineAdaptiveTransformer:
    """
    Performs test-time weight adaptation (TTT) anchored by an L2 pullback regularizer:
    Loss = TaskLoss + pullback * ||theta - theta_base||^2
    Prevents catastrophic forgetting while tuning the model to current network dynamics.
    """

    def __init__(self, base_model: nn.Module, lr: float = 3e-4, pullback: float = 5e-3):
        self.model = copy.deepcopy(base_model)
        self.model.eval()
        self.base_params = [p.clone().detach() for p in base_model.parameters()]
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.pullback = pullback
        self.recon_crit = nn.MSELoss()

    def adapt_step(self, seq_history: torch.Tensor, target_state: torch.Tensor, steps: int = 2) -> float:
        """Takes small SGD steps toward revealed telemetry."""
        self.model.train()
        total_loss = 0.0
        for _ in range(steps):
            self.optimizer.zero_grad()
            dyn_mean, dyn_logvar, atk_logits, mitre_logits, frac, last_attn = self.model(seq_history)
            pred_next = dyn_mean
            recon_loss = self.recon_crit(pred_next, target_state)

            
            # Pullback regularization to prevent model drift
            pullback_penalty = sum(
                ((p - b) ** 2).sum() for p, b in zip(self.model.parameters(), self.base_params)
            )
            loss = recon_loss + self.pullback * pullback_penalty
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        self.model.eval()
        return total_loss / steps


class RAMXPredictor:
    """
    Unified RAMX Predictor: Combines the StateTransformerWorldModel with
    Warmup Calibration, Test-Time Online Adaptation, and Episodic Memory.
    """

    def __init__(
        self,
        base_model: nn.Module,
        scaler_mean: np.ndarray,
        scaler_std: np.ndarray,
        warmup_steps: int = 15,
        context_skip_steps: int = 0,
        enable_ttt: bool = True,
    ):
        self.base_model = base_model
        self.scaler_mean = scaler_mean
        self.scaler_std = scaler_std
        self.warmup_steps = warmup_steps
        self.context_skip_steps = context_skip_steps
        self.calibrator = WarmupBaselineCalibrator(warmup_steps=warmup_steps)
        self.memory_bank = EpisodicMemoryBank()
        self.enable_ttt = enable_ttt
        self.adaptive_engine = OnlineAdaptiveTransformer(base_model) if enable_ttt else None
        self.step_count = 0

    def reset_stream(self) -> None:
        """Resets the calibrator for a new capture stream."""
        self.calibrator = WarmupBaselineCalibrator(warmup_steps=self.warmup_steps)
        self.step_count = 0

    def predict_state(
        self,
        raw_state_trajectory: np.ndarray,
        device: str = "cpu"
    ) -> Dict[str, any]:
        """
        Runs adaptive prediction on a state trajectory (L, D_state).
        Combines model output with local relative drift detection.
        """
        self.step_count += 1
        current_raw_state = raw_state_trajectory[-1]
        
        # 1. Update warmup baseline
        self.calibrator.update(current_raw_state)
        
        # 2. Compute local relative anomaly probability
        relative_anomaly = self.calibrator.get_relative_anomaly_score(current_raw_state)
        
        # 3. Model inference
        norm_seq = (raw_state_trajectory - self.scaler_mean) / self.scaler_std
        norm_seq = np.nan_to_num(norm_seq, nan=0.0)
        t_seq = torch.tensor(norm_seq, dtype=torch.float32).unsqueeze(0).to(device)

        active_model = self.adaptive_engine.model if self.adaptive_engine else self.base_model
        active_model.eval()
        with torch.no_grad():
            dyn_mean, dyn_logvar, atk_logits, mitre_logits, frac, last_attn = active_model(t_seq)
            raw_base_prob = float(torch.softmax(atk_logits, dim=-1)[0, 1].item())
            mitre_probs = torch.softmax(mitre_logits, dim=-1)[0].cpu().numpy()
            pred_stage = int(np.argmax(mitre_probs))
            frac_val = float(frac[0, 0].item())


        # 4. RAMX Fusion: Fuse base model probability with local relative drift.
        # During external context (context_skip_steps), learn baseline stats but
        # never raise alerts — fusion applies only to the live capture stream.
        in_context_gate = self.step_count <= self.context_skip_steps
        if self.calibrator.calibrated and not in_context_gate:
            fused_p_attack = float(max(raw_base_prob, 0.4 * raw_base_prob + 0.6 * relative_anomaly))
        else:
            fused_p_attack = float(raw_base_prob)

        # 5. Episodic Prototype Querying
        matches = self.memory_bank.query(norm_seq, k=1)
        if matches and matches[0][1] < 15.0:
            match_stage = matches[0][0]["stage"]
            if pred_stage == 0 and match_stage > 0:
                pred_stage = match_stage
                fused_p_attack = max(fused_p_attack, 0.65)

        return {
            "p_attack": float(np.clip(fused_p_attack, 0.0, 1.0)),
            "raw_p_attack": raw_base_prob,
            "relative_anomaly": relative_anomaly,
            "mitre_stage": pred_stage,
            "traffic_fraction": frac_val,
            "calibrated": self.calibrator.calibrated,
            "context_gated": in_context_gate,
            "ramx_version": RAMX_VERSION,
        }
