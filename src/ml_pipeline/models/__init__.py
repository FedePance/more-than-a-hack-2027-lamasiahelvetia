"""Model registry and factory for possession sequence models."""

from torch import nn

from src.ml_pipeline.models.bi_gru import PossessionBiGRU

SUPPORTED_ARCHITECTURES = ["bi_gru"]


def build_model(
    arch: str,
    input_dim: int = 14,
    num_classes: int = 3,
    **kwargs,
) -> nn.Module:
    """Build and instantiate a sequence model architecture.

    Args:
        arch: Architecture name ('bi_gru', etc.).
        input_dim: Number of input features (default: 14).
        num_classes: Target classes (default: 3: H, A, D).
        **kwargs: Additional model parameters (e.g. hidden_size, num_layers, dropout).

    Returns:
        Instantiated nn.Module.
    """
    clean_arch = arch.lower().strip()
    if clean_arch in ("bi_gru", "bigru", "gru"):
        return PossessionBiGRU(input_dim=input_dim, num_classes=num_classes, **kwargs)
    raise ValueError(
        f"Unsupported architecture '{arch}'. Available options: {SUPPORTED_ARCHITECTURES}"
    )


__all__ = ["SUPPORTED_ARCHITECTURES", "PossessionBiGRU", "build_model"]
