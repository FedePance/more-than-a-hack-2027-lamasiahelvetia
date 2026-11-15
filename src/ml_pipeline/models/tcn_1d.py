"""1D Dilated Temporal Convolutional Network (TCN) for possession sequence classification.

Features:
- Symmetric (non-causal, zero-phase-lag) dilated convolutions.
- GroupNorm and GELU activations for stable convergence and gradient flow.
- Exponential dilation progression: (1, 2, 4, 8, 16) with kernel size 5,
  providing ~250 frames (10.0 s at 25 fps) of bidirectional context.
- Residual skip connections preserving multi-scale temporal context.
- Standard (batch, seq_len, input_dim) -> (batch, seq_len, num_classes) interface.
"""

import torch
from torch import nn

DropoutLayer = getattr(nn, "Dropout1d", nn.Dropout)


class DilatedResidualBlock(nn.Module):
    """Symmetric dilated residual block with GroupNorm, GELU, and Dropout1d."""

    def __init__(
        self,
        channels: int,
        dilation: int,
        kernel_size: int = 5,
        dropout: float = 0.25,
    ):
        """Initialize DilatedResidualBlock.

        Args:
            channels: Number of hidden channels.
            dilation: Dilation factor (1, 2, 4, 8, 16, etc.).
            kernel_size: Convolution kernel size (default: 5).
            dropout: Dropout probability (default: 0.25).
        """
        super().__init__()
        # Symmetric padding so output sequence length matches seq_len exactly
        padding = dilation * (kernel_size - 1) // 2
        num_groups = 8 if channels % 8 == 0 else (4 if channels % 4 == 0 else 1)

        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.act1 = nn.GELU()
        self.drop1 = DropoutLayer(dropout)

        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.act2 = nn.GELU()
        self.drop2 = DropoutLayer(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual skip connection.

        Args:
            x: Input tensor of shape (batch, channels, seq_len).

        Returns:
            Output tensor of shape (batch, channels, seq_len).
        """
        residual = x
        h = self.drop1(self.act1(self.norm1(self.conv1(x))))
        h = self.drop2(self.act2(self.norm2(self.conv2(h))))
        return h + residual


class PossessionTCN1D(nn.Module):
    """1D Dilated Temporal Convolutional Network for per-frame possession classification (H, A, D)."""

    def __init__(
        self,
        input_dim: int = 14,
        hidden_size: int = 96,
        num_layers: int = 2,
        dropout: float = 0.25,
        num_classes: int = 3,
        **kwargs,
    ):
        """Initialize PossessionTCN1D.

        Args:
            input_dim: Number of input features per frame (default: 14).
            hidden_size: Hidden channel dimensionality (default: 96).
            num_layers: Number of stacked dilation cycles (default: 2).
            dropout: Dropout probability (default: 0.25).
            num_classes: Number of target states (default: 3: H, A, D).
            **kwargs: Extra parameters for factory compatibility.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_classes = num_classes

        num_groups = 8 if hidden_size % 8 == 0 else (4 if hidden_size % 4 == 0 else 1)

        # 1. Input projection from input_dim to hidden_size channels
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_dim, hidden_size, kernel_size=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=hidden_size),
            nn.GELU(),
        )

        # 2. Stack of symmetric dilated residual blocks with dilations (1, 2, 4, 8, 16)
        dilations = (1, 2, 4, 8, 16)
        blocks = []
        for _ in range(max(1, num_layers)):
            for d in dilations:
                blocks.append(
                    DilatedResidualBlock(
                        channels=hidden_size,
                        dilation=d,
                        kernel_size=5,
                        dropout=dropout,
                    )
                )
        self.blocks = nn.Sequential(*blocks)

        # 3. Per-frame classification head projecting (batch, hidden_size, seq_len)
        #    back to (batch, seq_len, num_classes)
        self.head = nn.Conv1d(hidden_size, num_classes, kernel_size=1)

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

        # Permute (batch, seq_len, input_dim) -> (batch, input_dim, seq_len) for Conv1d
        h = x.transpose(1, 2)
        h = self.input_proj(h)
        h = self.blocks(h)
        logits = self.head(h)  # (batch, num_classes, seq_len)
        logits = logits.transpose(1, 2).contiguous()  # (batch, seq_len, num_classes)

        if squeeze_batch:
            logits = logits.squeeze(0)

        return logits
