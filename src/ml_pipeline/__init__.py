"""Modular Machine Learning Pipeline for Ball Possession Identification.

Team: 'La MasIA Helvetia'
Phase 1: Possession Identification (More Than A Hack 2027)
"""

from src.ml_pipeline.dataset_open import (
    OpenGameData,
    OpenGameDataset,
    generate_ball_dropout_mask,
    load_open_game_data,
)
from src.ml_pipeline.features import (
    FEATURE_NAMES,
    MISSING_DIST_SENTINEL,
    N_FEATURES,
    extract_frame_features,
    extract_frame_features_from_tracking,
)
from src.ml_pipeline.models import PossessionBiGRU, build_model
from src.ml_pipeline.viterbi import codes_to_labels, viterbi_decode

__all__ = [
    "FEATURE_NAMES",
    "MISSING_DIST_SENTINEL",
    "N_FEATURES",
    "OpenGameData",
    "OpenGameDataset",
    "PossessionBiGRU",
    "build_model",
    "codes_to_labels",
    "extract_frame_features",
    "extract_frame_features_from_tracking",
    "generate_ball_dropout_mask",
    "load_open_game_data",
    "viterbi_decode",
]
