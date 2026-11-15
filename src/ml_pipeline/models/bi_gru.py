"""Bidirectional GRU architecture for sequence possession identification."""

import torch
from torch import nn


class ResidualMLPHead(nn.Module):
    """Classification head with residual projection for per-frame logits."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        num_classes: int = 3,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.ln1 = nn.LayerNorm(hidden_features)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, num_classes)
        self.skip = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass computing per-frame logits with residual shortcut.

        Args:
            x: Input tensor of shape (batch_size, seq_len, in_features).

        Returns:
            Logits of shape (batch_size, seq_len, num_classes).
        """
        residual = self.skip(x)
        h = self.drop(self.act(self.ln1(self.fc1(x))))
        return self.fc2(h) + residual


class PossessionBiGRU(nn.Module):
    """Bidirectional GRU model for per-frame ball possession classification (H, A, D)."""

    def __init__(
        self,
        input_dim: int = 14,
        hidden_size: int = 96,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.25,
    ):
        """Initialize PossessionBiGRU.

        Args:
            input_dim: Number of input features per frame (default: 14).
            hidden_size: GRU hidden state dimensionality (default: 96).
            num_layers: Number of stacked bidirectional GRU layers (default: 2).
            num_classes: Number of target states (default: 3: H, A, D).
            dropout: Dropout probability (default: 0.25).
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        # 1. Input projection and normalization
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 2. Multi-layer Bidirectional GRU
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 3. Residual MLP classification head (GRU bidirectional output is 2 * hidden_size)
        self.head = ResidualMLPHead(
            in_features=hidden_size * 2,
            hidden_features=hidden_size,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Feature tensor of shape (batch_size, seq_len, input_dim)
               or (seq_len, input_dim).

        Returns:
            Logits tensor of shape (batch_size, seq_len, num_classes).
        """
        squeeze_batch = False
        if x.ndim == 2:
            x = x.unsqueeze(0)
            squeeze_batch = True

        h = self.input_proj(x)
        out, _ = self.gru(h)
        logits = self.head(out)

        if squeeze_batch:
            logits = logits.squeeze(0)

        return logits
