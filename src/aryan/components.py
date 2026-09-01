"""
PRISM Shared Model Components
Reusable layers: positional encoding, attention, prediction heads.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding for sequence models."""

    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        return x + self.pe[:, : x.size(1), :]


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al.)."""

    def __init__(self, max_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class StateEmbedding(nn.Module):
    """Project raw state vector to model dimension with normalisation."""

    def __init__(self, d_state: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(d_state, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_state) -> (batch, seq_len, d_model)"""
        return self.dropout(self.norm(self.proj(x)))


class StatePredictionHead(nn.Module):
    """
    Predict next state as a Gaussian distribution: N(mu, sigma^2).
    Outputs mean and log-variance of the predicted next state.
    """

    def __init__(self, d_model: int, d_state: int, dropout: float = 0.3):
        super().__init__()
        self.mean_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_state),
        )
        self.logvar_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_state),
        )

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        h: (batch, d_model) — last hidden state
        Returns: (mean, logvar), each (batch, d_state)
        """
        return self.mean_head(h), self.logvar_head(h)


class ClassificationHead(nn.Module):
    """Generic classification head with dropout."""

    def __init__(
        self, d_model: int, num_classes: int, dropout: float = 0.3
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (batch, d_model) -> (batch, num_classes)"""
        return self.head(h)


class TemporalConv1DBlock(nn.Module):
    """
    Causal 1D Temporal Convolution Block.
    Extracts local temporal dynamics (e.g. packet bursts, inter-arrival variations,
    sequential port probing) across time windows with causal padding and residual connection.
    """

    def __init__(self, d_model: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size - 1  # Causal left padding
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=0,
        )
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        Returns: (batch, seq_len, d_model)
        """
        residual = x
        # (batch, d_model, seq_len)
        x_trans = x.transpose(1, 2)
        # Causal left padding along time dimension
        x_pad = F.pad(x_trans, (self.padding, 0))
        out = self.conv(x_pad)
        out = out.transpose(1, 2)  # (batch, seq_len, d_model)
        out = self.act(out)
        out = self.dropout(out)
        return self.norm(residual + out)


class RelativeTemporalBias(nn.Module):
    """
    Learnable Relative Temporal Attention Bias.
    Provides temporal inductive bias: how long ago an event happened (recency vs distant history).
    """

    def __init__(self, max_len: int = 128, n_heads: int = 8):
        super().__init__()
        self.max_len = max_len
        self.n_heads = n_heads
        # Relative distance bias: d = i - j for i >= j in causal attention
        self.bias_table = nn.Parameter(torch.zeros(n_heads, max_len))
        nn.init.trunc_normal_(self.bias_table, std=0.02)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Returns relative temporal bias matrix of shape (1, n_heads, seq_len, seq_len)
        or (n_heads * batch, seq_len, seq_len).
        """
        seq_len = min(seq_len, self.max_len)
        positions = torch.arange(seq_len, device=device)
        rel_dist = positions.unsqueeze(1) - positions.unsqueeze(0)  # (seq_len, seq_len), i - j
        rel_dist = rel_dist.clamp(min=0, max=self.max_len - 1)  # only non-negative for causal
        bias = self.bias_table[:, rel_dist]  # (n_heads, seq_len, seq_len)
        return bias


class TemporalAttentionPooling(nn.Module):
    """
    Temporal Sequence Attention Pooling with Gated Residual Highway.
    Aggregates full temporal context over lookback sequence L via learned attention queries,
    fusing historical threat precursors (e.g. stealth reconnaissance from 10 steps ago)
    with the instantaneous latest state h_L.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.query_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1, bias=False),
        )
        self.fuse = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, hidden_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        hidden_seq: (batch, seq_len, d_model)
        Returns:
            fused_h: (batch, d_model) — fused instantaneous + temporal context
            attn_weights: (batch, seq_len) — temporal attention distribution
        """
        scores = self.query_proj(hidden_seq)  # (batch, seq_len, 1)
        attn_weights = F.softmax(scores, dim=1)  # (batch, seq_len, 1)
        context = (hidden_seq * attn_weights).sum(dim=1)  # (batch, d_model)

        h_last = hidden_seq[:, -1, :]  # instantaneous latest state
        gate = torch.sigmoid(self.gate(h_last))  # adaptive temporal gate
        fused = self.norm(h_last + gate * self.fuse(context))
        return fused, attn_weights.squeeze(-1)


class FocalLoss(nn.Module):
    """
    Focal Cross-Entropy Loss for addressing extreme class imbalance in cyber attack data.
    Down-weights easy well-classified negative (benign) examples and focuses learning on rare attacks.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (batch, num_classes)
        targets: (batch,)
        """
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class RobustDynamicsLoss(nn.Module):
    """
    Robust Dynamics Loss combining Smooth L1 (Huber) next-state prediction
    with bounded variance calibration.
    Strictly positive and scale-stable, preventing feature scale distortion
    (e.g., from num_flows) and preventing negative loss collapse.
    """

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        state_loss = F.smooth_l1_loss(mean, target, beta=self.beta)
        logvar_clamped = logvar.clamp(-4.0, 2.0)
        sq_err = ((target - mean).detach()) ** 2
        var = logvar_clamped.exp()
        var_loss = 0.1 * ((sq_err / (var + 1e-4) + logvar_clamped).clamp(min=0.0, max=10.0).mean())
        return state_loss + var_loss


# Alias for backward compatibility
RobustGaussianNLL = RobustDynamicsLoss


class GaussianNLL(nn.Module):
    """Gaussian negative log-likelihood loss for state prediction."""

    def __init__(self, min_logvar: float = -10.0, max_logvar: float = 2.0):
        super().__init__()
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar

    def forward(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Gaussian NLL: 0.5 * (logvar + (target - mean)^2 / var).
        """
        logvar = logvar.clamp(self.min_logvar, self.max_logvar)
        var = logvar.exp()
        nll = 0.5 * (logvar + (target - mean) ** 2 / (var + 1e-8))
        return nll.mean()


class MultiTaskLoss(nn.Module):
    """
    Combined loss for world model training:
      L = lambda_d * L_dynamics + lambda_i * L_infiltration + lambda_m * L_mitre
    Uses RobustDynamicsLoss for dynamics and balanced CrossEntropy for classification.
    """

    def __init__(
        self,
        lambda_dynamics: float = 0.5,
        lambda_infiltration: float = 1.2,
        lambda_mitre: float = 1.0,
        binary_class_weights: Optional[torch.Tensor] = None,
        mitre_class_weights: Optional[torch.Tensor] = None,
        use_focal: bool = False,
    ):
        super().__init__()
        self.lambda_d = lambda_dynamics
        self.lambda_i = lambda_infiltration
        self.lambda_m = lambda_mitre

        self.dynamics_loss = RobustDynamicsLoss()
        if use_focal:
            self.infiltration_loss = FocalLoss(
                gamma=1.5, weight=binary_class_weights, label_smoothing=0.01
            )
            self.mitre_loss = FocalLoss(
                gamma=1.5, weight=mitre_class_weights, label_smoothing=0.01
            )
        else:
            self.infiltration_loss = nn.CrossEntropyLoss(
                weight=binary_class_weights, label_smoothing=0.01
            )
            self.mitre_loss = nn.CrossEntropyLoss(
                weight=mitre_class_weights, label_smoothing=0.01
            )

    def forward(
        self,
        pred_mean: torch.Tensor,
        pred_logvar: torch.Tensor,
        target_state: torch.Tensor,
        pred_binary: torch.Tensor,
        target_binary: torch.Tensor,
        pred_mitre: torch.Tensor,
        target_mitre: torch.Tensor,
    ) -> dict:
        """
        Compute multi-task loss.
        Returns dict with total loss and individual components.
        """
        l_dyn = self.dynamics_loss(pred_mean, pred_logvar, target_state)
        l_inf = self.infiltration_loss(pred_binary, target_binary)
        l_mit = self.mitre_loss(pred_mitre, target_mitre)

        total = self.lambda_d * l_dyn + self.lambda_i * l_inf + self.lambda_m * l_mit

        return {
            "total": total,
            "dynamics": l_dyn,
            "infiltration": l_inf,
            "mitre": l_mit,
        }


def generate_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Generate upper-triangular causal mask for Transformer."""
    mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
        diagonal=1,
    )
    return mask
