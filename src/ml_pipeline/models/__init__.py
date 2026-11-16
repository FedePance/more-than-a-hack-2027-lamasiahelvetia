"""Model registry and factory for possession sequence and tabular models."""

from typing import Any
from torch import nn

from src.ml_pipeline.models.bi_gru import PossessionBiGRU
from src.ml_pipeline.models.lgbm_model import PossessionLGBM, expand_temporal_tabular_features
from src.ml_pipeline.models.tcn_1d import PossessionTCN1D

SUPPORTED_ARCHITECTURES = ["bi_gru", "tcn_1d", "lgbm"]

# Accepted spellings (lower case) -> canonical architecture name
ARCH_ALIASES = {
    "bi_gru": "bi_gru",
    "bigru": "bi_gru",
    "gru": "bi_gru",
    "tcn_1d": "tcn_1d",
    "tcn1d": "tcn_1d",
    "tcn": "tcn_1d",
    "lgbm": "lgbm",
    "lightgbm": "lgbm",
}


def build_model(
    arch: str,
    input_dim: int = 14,
    num_classes: int = 3,
    **kwargs: Any,
) -> nn.Module | PossessionLGBM:
    """Build and instantiate a sequence or tabular model architecture.

    Args:
        arch: Architecture name or alias ('bi_gru', 'tcn_1d', 'lgbm', etc.).
        input_dim: Number of input features (default: 14).
        num_classes: Target classes (default: 3: H, A, D).
        **kwargs: Additional model parameters (e.g. hidden_size, num_layers, dropout).

    Returns:
        Instantiated nn.Module or PossessionLGBM.

    Raises:
        ValueError: If the architecture name is not recognized.
    """
    canonical_arch = ARCH_ALIASES.get(str(arch).lower().strip())
    if canonical_arch == "bi_gru":
        return PossessionBiGRU(input_dim=input_dim, num_classes=num_classes, **kwargs)
    if canonical_arch == "tcn_1d":
        return PossessionTCN1D(input_dim=input_dim, num_classes=num_classes, **kwargs)
    if canonical_arch == "lgbm":
        return PossessionLGBM(**kwargs)
    raise ValueError(
        f"Unsupported architecture '{arch}'. Available options: {SUPPORTED_ARCHITECTURES}"
    )


__all__ = [
    "ARCH_ALIASES",
    "SUPPORTED_ARCHITECTURES",
    "PossessionBiGRU",
    "PossessionTCN1D",
    "PossessionLGBM",
    "build_model",
    "expand_temporal_tabular_features",
]
