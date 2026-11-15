"""Model registry and factory for possession sequence and tabular models."""

from typing import Any
from torch import nn

from src.ml_pipeline.models.bi_gru import PossessionBiGRU
from src.ml_pipeline.models.lgbm_model import PossessionLGBM, expand_temporal_tabular_features
from src.ml_pipeline.models.tcn_1d import PossessionTCN1D

SUPPORTED_ARCHITECTURES = ["bi_gru", "tcn_1d", "lgbm"]


def build_model(
    arch: str,
    input_dim: int = 14,
    num_classes: int = 3,
    **kwargs: Any,
) -> nn.Module | PossessionLGBM:
    """Build and instantiate a sequence or tabular model architecture.

    Args:
        arch: Architecture name ('bi_gru', 'tcn_1d', 'lgbm', etc.).
        input_dim: Number of input features (default: 14).
        num_classes: Target classes (default: 3: H, A, D).
        **kwargs: Additional model parameters (e.g. hidden_size, num_layers, dropout).

    Returns:
        Instantiated nn.Module or PossessionLGBM.
    """
    clean_arch = arch.lower().strip()
    if clean_arch in ("bi_gru", "bigru", "gru"):
        return PossessionBiGRU(input_dim=input_dim, num_classes=num_classes, **kwargs)
    if clean_arch in ("tcn_1d", "tcn1d", "tcn"):
        return PossessionTCN1D(input_dim=input_dim, num_classes=num_classes, **kwargs)
    if clean_arch in ("lgbm", "lightgbm"):
        return PossessionLGBM(**kwargs)
    raise ValueError(
        f"Unsupported architecture '{arch}'. Available options: {SUPPORTED_ARCHITECTURES}"
    )


__all__ = [
    "SUPPORTED_ARCHITECTURES",
    "PossessionBiGRU",
    "PossessionTCN1D",
    "PossessionLGBM",
    "build_model",
    "expand_temporal_tabular_features",
]
