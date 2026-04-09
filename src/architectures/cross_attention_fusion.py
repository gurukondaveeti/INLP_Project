# Cross-Branch Attention Fusion Architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ─────────────────────────────────────────────────────────────────
# TOKEN EMBEDDING
# Projects a flat feature vector into a sequence of token embeddings
# Input:  (B, num_features)
# Output: (B, num_features, d_model)   — each feature is one "token"
# ─────────────────────────────────────────────────────────────────
class TokenEmbedding(nn.Module):
    def __init__(self, num_features: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        # Each feature gets its own learnable linear projection
        self.proj = nn.Linear(1, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F)  →  unsqueeze to (B, F, 1)  →  project to (B, F, d_model)
        x = x.unsqueeze(-1)          # (B, F, 1)
        x = self.proj(x)             # (B, F, d_model)
        x = self.norm(x)
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────────
# CROSS-ATTENTION LAYER
# Query tokens from branch A attend to Key/Value tokens from branch B
# ─────────────────────────────────────────────────────────────────
class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,          # expects (B, seq, d_model)
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,    # (B, Q_len, d_model)  — the "questioner" branch
        context: torch.Tensor,  # (B, K_len, d_model)  — the "answerer" branch
    ) -> torch.Tensor:
        attn_out, _ = self.attn(query=query, key=context, value=context)
        # Residual connection + LayerNorm
        return self.norm(query + self.dropout(attn_out))


# ─────────────────────────────────────────────────────────────────
# SELF-ATTENTION REFINEMENT LAYER
# After cross-attention enrichment, each branch refines itself
# ─────────────────────────────────────────────────────────────────
class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention sub-layer
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + self.dropout(attn_out))
        # Feed-forward sub-layer
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


# ─────────────────────────────────────────────────────────────────
# RESIDUAL CLASSIFIER HEAD
# Deep MLP with skip connections for stable deep learning
# ─────────────────────────────────────────────────────────────────
class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class ClassifierHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.res1 = ResidualBlock(hidden_dim, dropout)
        self.res2 = ResidualBlock(hidden_dim, dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.res1(x)
        x = self.res2(x)
        return self.head(x)


# ─────────────────────────────────────────────────────────────────
# MAIN MODEL: Cross-Branch Attention Fusion
# ─────────────────────────────────────────────────────────────────
class CrossAttentionFusionMLP(nn.Module):
    """
    2-Branch Cross-Attention Fusion for MGT Detection.

    Pipeline:
        Stylo  (B, 4)  ──► TokenEmbed ──► stylo_tokens (B, 4, d_model)
        Perp   (B, 12) ──► TokenEmbed ──► perp_tokens  (B, 12, d_model)
                          CrossAttn: stylo queries perp  ──► enriched_stylo
                          CrossAttn: perp  queries stylo ──► enriched_perp
                          SelfAttn refinement on each
                          MeanPool: (B, d_model) each
                          Concat:   (B, 2*d_model)
                          ClassifierHead → logit (B, 1)
    """
    def __init__(
        self,
        stylo_dim: int,
        perp_dim: int,
        d_model: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.stylo_dim = stylo_dim
        self.perp_dim = perp_dim
        self.d_model = d_model

        # Token Embeddings
        self.stylo_embed = TokenEmbedding(stylo_dim, d_model, dropout)
        self.perp_embed = TokenEmbedding(perp_dim, d_model, dropout)

        # Stacked Cross-Attention + Self-Attention layers
        self.cross_layers = nn.ModuleList([
            nn.ModuleDict({
                "stylo_cross": CrossAttentionLayer(d_model, num_heads, dropout),  # stylo attends perp
                "perp_cross": CrossAttentionLayer(d_model, num_heads, dropout),   # perp attends stylo
                "stylo_self": SelfAttentionLayer(d_model, num_heads, dropout),
                "perp_self": SelfAttentionLayer(d_model, num_heads, dropout),
            })
            for _ in range(num_layers)
        ])

        # Classifier
        fused_dim = d_model * 2
        self.classifier = ClassifierHead(fused_dim, hidden_dim, dropout)

        self._init_weights()
        self._print_params()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _print_params(self):
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Total params:     {total:,}")
        print(f"  Trainable params: {train:,}")

    def forward(self, stylo: torch.Tensor, perp: torch.Tensor) -> torch.Tensor:
        """
        Args:
            stylo: (B, stylo_dim)  — stylometric features
            perp:  (B, perp_dim)   — perplexity features
        Returns:
            logits: (B, 1)
        """
        # ── 1. Token Embeddings ───────────────────────────────────
        s = self.stylo_embed(stylo)   # (B, 4, d_model)
        p = self.perp_embed(perp)     # (B, 12, d_model)

        # ── 2. Stacked Cross-Attention + Self-Attention ───────────
        for layer in self.cross_layers:
            # Cross-attention (bidirectional)
            s_new = layer["stylo_cross"](query=s, context=p)   # stylo asks perp
            p_new = layer["perp_cross"](query=p, context=s)    # perp asks stylo
            s, p = s_new, p_new

            # Self-attention refinement
            s = layer["stylo_self"](s)
            p = layer["perp_self"](p)

        # ── 3. Mean Pooling over token dimension ──────────────────
        s_pooled = s.mean(dim=1)    # (B, d_model)
        p_pooled = p.mean(dim=1)    # (B, d_model)

        # ── 4. Fusion + Classifier ────────────────────────────────
        fused = torch.cat([s_pooled, p_pooled], dim=-1)   # (B, 2*d_model)
        return self.classifier(fused)                      # (B, 1)
