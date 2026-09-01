"""
PRISM - Core Transformer World Model
Learns P(S_{t+1} | S_{t-L+1}, ..., S_t) via causal self-attention.
Three output heads: state dynamics, infiltration binary, MITRE stage.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.aryan.components import (
    StateEmbedding,
    LearnablePositionalEncoding,
    SinusoidalPositionalEncoding,
    TemporalConv1DBlock,
    TemporalAttentionPooling,
    StatePredictionHead,
    ClassificationHead,
    MultiTaskLoss,
    generate_causal_mask,
)
from src.aryan.constants import NUM_MITRE_STAGES

logger = logging.getLogger("prism.models.world_model")


class TemporalTransformerWorldModel(nn.Module):
    """
    Temporal Transformer World Model for network state transition dynamics.
    Learns P(S_{t+1} | S_{t-L+1}, ..., S_t) via causal multi-head temporal self-attention,
    causal 1D temporal convolutions, temporal attention pooling, and residual state dynamics.

    Architecture:
        Input:  [S_{t-L+1}, ..., S_t]  shape (B, L, D_state)
        1. State Embedding: Linear(D_state -> D_model) + LayerNorm
        2. Causal 1D Temporal Convolution (extracts local burst & sequence shapes)
        3. Positional Encoding (learnable or sinusoidal)
        4. Causal Transformer Encoder (N layers, H heads, Pre-LN)
        5. Temporal Attention Pooling (fuses sequence context with instantaneous state)
        6. Residual State Dynamics Head: S_{t+1} = S_t + Delta S_t
        7. Infiltration Binary Head: h_fused -> P(attack | trajectory)
        8. MITRE Attack Stage Head: h_fused -> P(stage | trajectory)
    """

    def __init__(
        self,
        d_state: int = 110,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        lookback: int = 20,
        dropout: float = 0.1,
        head_dropout: float = 0.3,
        num_mitre_stages: int = NUM_MITRE_STAGES,
        pos_encoding: str = "learnable",  # "learnable" | "sinusoidal"
        residual_dynamics: bool = True,
    ):
        super().__init__()
        self.d_state = d_state
        self.d_model = d_model
        self.lookback = lookback
        self.num_mitre_stages = num_mitre_stages
        self.residual_dynamics = residual_dynamics

        # 1. Input embedding
        self.embedding = StateEmbedding(d_state, d_model, dropout)

        # 2. Causal 1D Temporal Convolution (local inter-window dynamics)
        self.temporal_conv = TemporalConv1DBlock(d_model, kernel_size=3, dropout=dropout)

        # 3. Positional encoding
        if pos_encoding == "learnable":
            self.pos_enc = LearnablePositionalEncoding(
                max_len=lookback + 64, d_model=d_model
            )
        else:
            self.pos_enc = SinusoidalPositionalEncoding(
                max_len=lookback + 64, d_model=d_model, dropout=dropout
            )

        # 4. Causal Transformer Encoder (Pre-LN for stability)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )

        # 5. Temporal Attention Pooling
        self.temporal_pool = TemporalAttentionPooling(d_model, dropout=dropout)

        # 6. Residual State Dynamics Head
        self.state_head = StatePredictionHead(d_model, d_state, head_dropout)

        # 7. Multi-task classification heads
        self.infiltration_head = ClassificationHead(d_model, 2, head_dropout)
        self.mitre_head = ClassificationHead(
            d_model, num_mitre_stages, head_dropout
        )

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "TemporalTransformerWorldModel: d_state=%d, d_model=%d, "
            "layers=%d, heads=%d, params=%.2fM, residual_dynamics=%s",
            d_state, d_model, n_layers, n_heads, n_params / 1e6, residual_dynamics,
        )

    def forward(
        self,
        state_seq: torch.Tensor,
        return_attention: bool = False,
    ) -> dict:
        """
        Forward pass.

        Parameters
        ----------
        state_seq : torch.Tensor, shape (B, L, D_state)
            Sequence of past network states.
        return_attention : bool
            If True, extract and return attention weights from last layer.

        Returns
        -------
        dict with keys:
            pred_state_mean       : (B, D_state) — predicted next state mean
            pred_state_logvar     : (B, D_state) — predicted next state log-variance
            pred_binary           : (B, 2)       — infiltration logits
            pred_mitre            : (B, N_stage) — MITRE stage logits
            hidden                : (B, D_model) — fused representation (for SHAP/viz)
            hidden_seq            : (B, L, D_model) — full sequence representations
            attention_weights     : (B, L, L) or None
            temporal_pool_weights : (B, L)
        """
        B, L, _ = state_seq.shape
        device = state_seq.device

        # 1. Embed & local temporal conv
        x = self.embedding(state_seq)            # (B, L, D_model)
        x = self.pos_enc(x)                      # (B, L, D_model)
        x = self.temporal_conv(x)                # (B, L, D_model)

        # 2. Causal temporal mask: position t only attends to <= t
        causal_mask = generate_causal_mask(L, device)  # (L, L)

        # 3. Causal Transformer Encoder
        hidden_seq = self.transformer(
            x, mask=causal_mask, is_causal=True
        )                                        # (B, L, D_model)

        # 4. Temporal Attention Pooling (fuses trajectory history + latest state)
        h_fused, pool_weights = self.temporal_pool(hidden_seq)

        # 5. Prediction heads
        delta_mean, pred_logvar = self.state_head(h_fused)

        # Residual dynamics: S_{t+1} = S_t + Delta S_t
        if self.residual_dynamics:
            s_t = state_seq[:, -1, :]
            pred_mean = s_t + delta_mean
        else:
            pred_mean = delta_mean

        pred_binary = self.infiltration_head(h_fused)
        pred_mitre = self.mitre_head(h_fused)

        out = {
            "pred_state_mean": pred_mean,
            "pred_state_logvar": pred_logvar,
            "pred_binary": pred_binary,
            "pred_mitre": pred_mitre,
            "hidden": h_fused,
            "hidden_seq": hidden_seq,
            "attention_weights": None,
            "temporal_pool_weights": pool_weights,
        }

        if return_attention:
            out["attention_weights"] = self._extract_attention(
                x, causal_mask, device
            )

        return out

    def predict_infiltration_prob(self, state_seq: torch.Tensor) -> torch.Tensor:
        """Convenience: return P(attack) scalar per batch item."""
        with torch.no_grad():
            out = self.forward(state_seq)
        return torch.softmax(out["pred_binary"], dim=-1)[:, 1]

    def sample_next_state(
        self,
        state_seq: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Sample (or take mean of) the predicted next state distribution.
        Used for K-step rollout.
        """
        with torch.no_grad():
            out = self.forward(state_seq)
        mean = out["pred_state_mean"]
        if deterministic:
            return mean
        logvar = out["pred_state_logvar"].clamp(-8.0, 2.0)
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mean + eps * std

    # ------------------------------------------------------------------
    # Attention extraction (for explainability)
    # ------------------------------------------------------------------
    def _extract_attention(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Re-run the last Transformer layer and extract attention weights.
        Returns (B, L, L) averaged over heads.
        """
        last_layer = self.transformer.layers[-1]
        attn_output, attn_weights = last_layer.self_attn(
            x, x, x,
            attn_mask=mask.float().masked_fill(mask, float("-inf")),
            need_weights=True,
            average_attn_weights=True,
        )
        return attn_weights  # (B, L, L)

    def _init_weights(self):
        """Initialize weights with Xavier and small residual head weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Initialize delta prediction head to near-zero for smooth residual start
        if hasattr(self.state_head, "mean_head"):
            last_linear = self.state_head.mean_head[-1]
            if isinstance(last_linear, nn.Linear):
                nn.init.normal_(last_linear.weight, std=1e-3)
                if last_linear.bias is not None:
                    nn.init.zeros_(last_linear.bias)


# Backward-compatible alias for existing code, imports, and checkpoints
StateTransformerWorldModel = TemporalTransformerWorldModel


class LSTMWorldModel(nn.Module):
    """
    LSTM-based World Model variant.
    Same input/output contract as StateTransformerWorldModel.
    """

    def __init__(
        self,
        d_state: int = 110,
        d_model: int = 256,
        lstm_layers: int = 2,
        lookback: int = 20,
        dropout: float = 0.1,
        head_dropout: float = 0.3,
        num_mitre_stages: int = NUM_MITRE_STAGES,
    ):
        super().__init__()
        self.d_state = d_state
        self.d_model = d_model
        self.lookback = lookback

        self.embedding = StateEmbedding(d_state, d_model, dropout)

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=False,  # causal: no look-ahead
        )

        self.state_head = StatePredictionHead(d_model, d_state, head_dropout)
        self.infiltration_head = ClassificationHead(d_model, 2, head_dropout)
        self.mitre_head = ClassificationHead(d_model, num_mitre_stages, head_dropout)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "LSTMWorldModel: d_state=%d, d_model=%d, "
            "lstm_layers=%d, params=%.2fM",
            d_state, d_model, lstm_layers, n_params / 1e6,
        )

    def forward(
        self,
        state_seq: torch.Tensor,
        hidden_state: Optional[tuple] = None,
        return_attention: bool = False,
    ) -> dict:
        """
        Parameters
        ----------
        state_seq    : (B, L, D_state)
        hidden_state : optional LSTM (h, c) for stateful inference
        """
        x = self.embedding(state_seq)                # (B, L, D_model)
        lstm_out, (h_n, c_n) = self.lstm(x, hidden_state)  # (B, L, D_model)

        h_t = lstm_out[:, -1, :]                     # (B, D_model)

        pred_mean, pred_logvar = self.state_head(h_t)
        pred_binary = self.infiltration_head(h_t)
        pred_mitre = self.mitre_head(h_t)

        return {
            "pred_state_mean": pred_mean,
            "pred_state_logvar": pred_logvar,
            "pred_binary": pred_binary,
            "pred_mitre": pred_mitre,
            "hidden": h_t,
            "hidden_seq": lstm_out,
            "lstm_state": (h_n, c_n),
            "attention_weights": None,
        }

    def predict_infiltration_prob(self, state_seq: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.forward(state_seq)
        return torch.softmax(out["pred_binary"], dim=-1)[:, 1]

    def sample_next_state(
        self, state_seq: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        with torch.no_grad():
            out = self.forward(state_seq)
        mean = out["pred_state_mean"]
        if deterministic:
            return mean
        logvar = out["pred_state_logvar"].clamp(-10.0, 2.0)
        std = (0.5 * logvar).exp()
        return mean + torch.randn_like(std) * std


def build_world_model(cfg) -> nn.Module:
    """
    Factory: build the world model specified in config.

    Parameters
    ----------
    cfg : ModelConfig dataclass or dict-like
    """
    arch = getattr(cfg, "architecture", "transformer").lower()

    if arch in ("transformer", "temporal_transformer", "temporal"):
        return TemporalTransformerWorldModel(
            d_state=cfg.d_state,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            lookback=getattr(cfg, "lookback", 20),
            dropout=cfg.dropout,
            head_dropout=cfg.head_dropout,
            residual_dynamics=getattr(cfg, "residual_dynamics", True),
        )
    elif arch == "lstm":
        return LSTMWorldModel(
            d_state=cfg.d_state,
            d_model=cfg.d_model,
            lstm_layers=cfg.lstm_layers,
            lookback=getattr(cfg, "lookback", 20),
            dropout=cfg.dropout,
            head_dropout=cfg.head_dropout,
        )
    elif arch == "gnn":
        from src.models.gnn_model import GraphWorldModel
        return GraphWorldModel(
            d_node=cfg.d_state,
            d_graph=cfg.d_graph,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            gnn_type=cfg.gnn_type,
            gnn_layers=cfg.gnn_layers,
            dropout=cfg.dropout,
            head_dropout=cfg.head_dropout,
        )
    elif arch == "latent":
        from src.models.latent_dynamics import LatentDynamicsWorldModel
        return LatentDynamicsWorldModel(
            d_state=cfg.d_state,
            d_latent=cfg.d_latent,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
            head_dropout=cfg.head_dropout,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")


def save_checkpoint(
    model: nn.Module,
    optimizer,
    epoch: int,
    metrics: dict,
    path: str,
) -> None:
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "model_class": model.__class__.__name__,
        },
        path,
    )
    logger.info("Saved checkpoint -> %s (epoch %d)", path, epoch)


def load_checkpoint(
    model: nn.Module,
    path: str,
    optimizer=None,
    device: str = "cpu",
) -> dict:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    logger.info(
        "Loaded checkpoint from %s (epoch %d)", path, ckpt.get("epoch", -1)
    )
    return ckpt
