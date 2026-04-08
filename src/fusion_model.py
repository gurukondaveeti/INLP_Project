"""
fusion_model.py
───────────────
Anti-Overshadowing Multimodal Fusion MLP for MGT Detection.

Architecture:
    3 branches (Stylometric 11-D, Perplexity 12-D, Semantic 64-D)
    → Independent LayerNorm per branch
    → Learnable sigmoid gates (perplexity gate initialized LOW)
    → Gradient scaling on perplexity branch (0.1x)
    → Modality dropout on perplexity branch (40%)
    → Concatenation → Primary MLP Head
    → Auxiliary MLP Head (stylo + semantic only)

Author: Fusion Team — SemEval-2024 Task 8A
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Tuple, Optional


# ══════════════════════════════════════════════════════════════
# 1. GRADIENT SCALER (Custom Autograd)
# ══════════════════════════════════════════════════════════════
class _GradientScaler(Function):
    """Scale gradients by a constant factor during backward pass.
    Forward pass: identity. Backward pass: grad * scale."""

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


def scale_gradient(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Apply gradient scaling: forward=identity, backward=scale*grad."""
    return _GradientScaler.apply(x, scale)


# ══════════════════════════════════════════════════════════════
# 2. RESIDUAL MLP BLOCK
# ══════════════════════════════════════════════════════════════
class MLPBlock(nn.Module):
    """Linear → LayerNorm → GELU → Dropout, with optional residual."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3,
                 use_residual: bool = False):
        super().__init__()
        self.use_residual = use_residual and (in_dim == out_dim)
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.drop(self.act(self.norm(self.linear(x))))
        if self.use_residual:
            out = out + x
        return out


# ══════════════════════════════════════════════════════════════
# 3. BRANCH PROJECTION (optional expansion for tiny branches)
# ══════════════════════════════════════════════════════════════
class BranchProjection(nn.Module):
    """Project a small-dim branch into a richer representation space."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ══════════════════════════════════════════════════════════════
# 4. MAIN FUSION MODEL
# ══════════════════════════════════════════════════════════════
class FusionMLP(nn.Module):
    """
    Anti-Overshadowing Multimodal Fusion MLP.

    Defenses against perplexity domination:
        1. Independent LayerNorm per branch
        2. Modality Dropout on perplexity (p=0.40)
        3. Gradient Scaling on perplexity (0.1x)
        4. Learnable Gating (perplexity gate init=0.12)
        5. Auxiliary Loss Head (stylo + semantic only)
    """

    def __init__(
        self,
        stylo_dim: int = 11,
        perp_dim: int = 12,
        sem_dim: int = 64,
        hidden_dims: Tuple[int, ...] = (256, 128, 64),
        dropout_rates: Tuple[float, ...] = (0.4, 0.3, 0.2),
        perp_modality_dropout: float = 0.40,
        perp_grad_scale: float = 0.10,
        perp_gate_init: float = 0.12,
        stylo_gate_init: float = 0.50,
        sem_gate_init: float = 0.50,
        use_branch_projection: bool = True,
        branch_proj_dim: int = 48,
    ):
        super().__init__()

        self.perp_modality_dropout = perp_modality_dropout
        self.perp_grad_scale = perp_grad_scale
        self.use_branch_projection = use_branch_projection

        # ── Branch Normalization ───────────────────────────────
        self.stylo_norm = nn.LayerNorm(stylo_dim)
        self.perp_norm = nn.LayerNorm(perp_dim)
        self.sem_norm = nn.LayerNorm(sem_dim)

        # ── Optional Branch Projections ────────────────────────
        # Project smaller branches (11-D, 12-D) into richer space
        if use_branch_projection:
            self.stylo_proj = BranchProjection(stylo_dim, branch_proj_dim)
            self.perp_proj = BranchProjection(perp_dim, branch_proj_dim)
            self.sem_proj = BranchProjection(sem_dim, branch_proj_dim)
            effective_stylo = branch_proj_dim
            effective_perp = branch_proj_dim
            effective_sem = branch_proj_dim
        else:
            self.stylo_proj = nn.Identity()
            self.perp_proj = nn.Identity()
            self.sem_proj = nn.Identity()
            effective_stylo = stylo_dim
            effective_perp = perp_dim
            effective_sem = sem_dim

        # ── Learnable Gates ────────────────────────────────────
        # Initialized via sigmoid inverse so that sigmoid(param) = desired_init
        def _gate_param(init_val):
            logit = torch.log(torch.tensor(init_val / (1.0 - init_val)))
            return nn.Parameter(logit.clone())

        self.gate_stylo = _gate_param(stylo_gate_init)
        self.gate_perp = _gate_param(perp_gate_init)
        self.gate_sem = _gate_param(sem_gate_init)

        # ── Primary MLP Head (all 3 branches) ─────────────────
        fusion_dim = effective_stylo + effective_perp + effective_sem
        layers = []
        in_d = fusion_dim
        for i, (h_dim, drop) in enumerate(zip(hidden_dims, dropout_rates)):
            layers.append(MLPBlock(in_d, h_dim, dropout=drop, use_residual=(i > 0)))
            in_d = h_dim
        self.primary_mlp = nn.Sequential(*layers)
        self.primary_head = nn.Linear(in_d, 1)

        # ── Auxiliary MLP Head (stylo + semantic ONLY) ─────────
        aux_fusion_dim = effective_stylo + effective_sem
        aux_layers = []
        in_d = aux_fusion_dim
        aux_hidden = (128, 64)
        aux_drops = (0.3, 0.2)
        for i, (h_dim, drop) in enumerate(zip(aux_hidden, aux_drops)):
            aux_layers.append(MLPBlock(in_d, h_dim, dropout=drop, use_residual=(i > 0)))
            in_d = h_dim
        self.aux_mlp = nn.Sequential(*aux_layers)
        self.aux_head = nn.Linear(in_d, 1)

        # ── Weight Initialization ─────────────────────────────
        self._init_weights()

    def _init_weights(self):
        """Kaiming init for Linear layers, constant for LayerNorm."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        stylo: torch.Tensor,    # (B, stylo_dim)
        perp: torch.Tensor,     # (B, perp_dim)
        sem: torch.Tensor,      # (B, sem_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        primary_logits : (B, 1) — prediction from ALL branches
        aux_logits     : (B, 1) — prediction from stylo + semantic ONLY
        """
        # ── 1. Branch Normalization ────────────────────────────
        s = self.stylo_norm(stylo)
        p = self.perp_norm(perp)
        e = self.sem_norm(sem)

        # ── 2. Gradient Scaling on perplexity ──────────────────
        p = scale_gradient(p, self.perp_grad_scale)

        # ── 3. Modality Dropout on perplexity ──────────────────
        if self.training:
            mask = torch.bernoulli(
                torch.full((p.size(0), 1), 1.0 - self.perp_modality_dropout,
                           device=p.device)
            )
            # Scale up surviving values to maintain expected magnitude
            p = p * mask / (1.0 - self.perp_modality_dropout + 1e-8)

        # ── 4. Branch Projections ──────────────────────────────
        s = self.stylo_proj(s)
        p = self.perp_proj(p)
        e = self.sem_proj(e)

        # ── 5. Learnable Gating ────────────────────────────────
        g_s = torch.sigmoid(self.gate_stylo)
        g_p = torch.sigmoid(self.gate_perp)
        g_e = torch.sigmoid(self.gate_sem)

        s_gated = s * g_s
        p_gated = p * g_p
        e_gated = e * g_e

        # ── 6. Fusion → Primary Head ──────────────────────────
        fused = torch.cat([s_gated, p_gated, e_gated], dim=-1)
        primary_logits = self.primary_head(self.primary_mlp(fused))

        # ── 7. Auxiliary Head (no perplexity) ──────────────────
        fused_aux = torch.cat([s_gated, e_gated], dim=-1)
        aux_logits = self.aux_head(self.aux_mlp(fused_aux))

        return primary_logits, aux_logits

    def get_gate_values(self) -> dict:
        """Return current gate activations (for monitoring)."""
        return {
            "gate_stylo": torch.sigmoid(self.gate_stylo).item(),
            "gate_perp": torch.sigmoid(self.gate_perp).item(),
            "gate_sem": torch.sigmoid(self.gate_sem).item(),
        }

    def get_fusion_dim(self) -> int:
        """Return total fusion dimension."""
        if self.use_branch_projection:
            return sum(
                p.net[0].out_features
                for p in [self.stylo_proj, self.perp_proj, self.sem_proj]
            )
        return (
            self.stylo_norm.normalized_shape[0]
            + self.perp_norm.normalized_shape[0]
            + self.sem_norm.normalized_shape[0]
        )


# ══════════════════════════════════════════════════════════════
# 5. COMBINED LOSS FUNCTION
# ══════════════════════════════════════════════════════════════
class FusionLoss(nn.Module):
    """
    Combined loss:
        total = α * BCE(primary) + (1-α) * BCE(auxiliary)

    The auxiliary loss forces the stylometric + semantic branches to be
    independently discriminative, preventing perplexity overshadowing.
    """

    def __init__(
        self,
        primary_weight: float = 0.70,
        label_smoothing: float = 0.05,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.alpha = primary_weight
        self.label_smoothing = label_smoothing

        # BCEWithLogitsLoss with optional class weighting
        self.primary_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.aux_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(
        self,
        primary_logits: torch.Tensor,  # (B, 1)
        aux_logits: torch.Tensor,      # (B, 1)
        labels: torch.Tensor,          # (B,) long
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (total_loss, primary_loss, aux_loss)
        """
        targets = labels.float().unsqueeze(1)

        # Label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        loss_primary = self.primary_criterion(primary_logits, targets)
        loss_aux = self.aux_criterion(aux_logits, targets)

        total = self.alpha * loss_primary + (1.0 - self.alpha) * loss_aux

        return total, loss_primary, loss_aux
