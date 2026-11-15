"""Symmetric rolling multi-scale temporal tabular feature expansion and LightGBM model.

Transforms the canonical 14-dimensional feature representations into a rich
multi-scale temporal feature matrix using fast vectorized scipy.ndimage 1D filters
and wraps LightGBM multiclass classifier for ball possession identification.
"""

from pathlib import Path
from typing import Any

import joblib
import lightgbm
import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d

# Key distance and speed feature column indices in the canonical 14 features:
# 0: min_dist_home, 1: min_dist_away, 2: dist_diff, 4: ball_speed,
# 7: closest_home_speed, 8: closest_away_speed, 9: outfield_mean_speed, 10: outfield_p80_speed
KEY_DIST_SPEED_INDICES = [0, 1, 2, 4, 7, 8, 9, 10]
TEMPORAL_KERNEL_SIZES = (13, 25, 51, 125)  # ~0.5s, 1.0s, 2.0s, 5.0s at 25 fps
SHORT_DIFF_LAG = 12  # ~0.48s at 25 fps


def expand_temporal_tabular_features(base_features: np.ndarray) -> np.ndarray:
    """Expand canonical 14-dimensional features into multi-scale temporal representations.

    Uses fast vectorized scipy.ndimage filters (uniform_filter1d, minimum_filter1d,
    maximum_filter1d) with symmetric centered kernels of sizes (13, 25, 51, 125)
    frames (~0.5s, 1.0s, 2.0s, 5.0s at 25 fps) to compute:
    1. Rolling means across all 14 features.
    2. Rolling standard deviations (sqrt(max(0, E[x^2] - E[x]^2))) across all 14 features.
    3. Rolling min and max across key distance and speed features.
    4. Short-term temporal differences (x[t] - x[t-12] and x[t+12] - x[t]).

    Args:
        base_features: Canonical 14-dimensional float32 feature array of shape (n_frames, 14).

    Returns:
        Expanded 2D numpy array of shape (n_frames, n_expanded_features), dtype float32.
    """
    base = np.asarray(base_features, dtype=np.float32)
    n_frames, n_feats = base.shape
    if n_frames == 0:
        return np.zeros((0, 0), dtype=np.float32)

    feature_blocks: list[np.ndarray] = [base]

    # Pre-select key distance and speed features for min/max filtering
    key_feats = base[:, KEY_DIST_SPEED_INDICES]

    # Precompute squared features for rolling variance
    base_squared = base * base

    # Symmetric centered multi-scale filters
    for k in TEMPORAL_KERNEL_SIZES:
        kernel_size = min(k, n_frames)
        if kernel_size % 2 == 0:
            kernel_size = max(1, kernel_size - 1)

        # 1. Rolling mean: E[x]
        mean_k = uniform_filter1d(base, size=kernel_size, axis=0, mode="nearest")
        feature_blocks.append(mean_k.astype(np.float32))

        # 2. Rolling standard deviation: sqrt(max(0, E[x^2] - E[x]^2))
        mean_sq_k = uniform_filter1d(base_squared, size=kernel_size, axis=0, mode="nearest")
        var_k = np.maximum(0.0, mean_sq_k - (mean_k * mean_k))
        std_k = np.sqrt(var_k)
        feature_blocks.append(std_k.astype(np.float32))

        # 3. Rolling min and max on key distance and speed features
        min_k = minimum_filter1d(key_feats, size=kernel_size, axis=0, mode="nearest")
        max_k = maximum_filter1d(key_feats, size=kernel_size, axis=0, mode="nearest")
        feature_blocks.append(min_k.astype(np.float32))
        feature_blocks.append(max_k.astype(np.float32))

    # 4. Short-term temporal differences: x[t] - x[t-12] and x[t+12] - x[t] (~0.48s at 25 fps)
    lag = SHORT_DIFF_LAG
    if n_frames > lag:
        padded = np.pad(base, ((lag, lag), (0, 0)), mode="edge")
        diff_past = padded[lag:-lag] - padded[:-2 * lag]
        diff_future = padded[2 * lag:] - padded[lag:-lag]
    else:
        diff_past = np.zeros_like(base)
        diff_future = np.zeros_like(base)

    feature_blocks.append(diff_past.astype(np.float32))
    feature_blocks.append(diff_future.astype(np.float32))

    expanded = np.ascontiguousarray(np.concatenate(feature_blocks, axis=1), dtype=np.float32)
    return np.nan_to_num(expanded, nan=0.0, posinf=50.0, neginf=-50.0)


class PossessionLGBM:
    """LightGBM tabular classifier wrapper for possession identification."""

    def __init__(
        self,
        n_estimators: int = 450,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        subsample_for_bin: int = 50000,
        colsample_bytree: float = 0.8,
        subsample: float = 0.8,
        subsample_freq: int = 1,
        min_child_samples: int = 50,
        n_jobs: int = 4,
        random_state: int = 42,
        verbose: int = -1,
        **kwargs: Any,
    ):
        self.params: dict[str, Any] = {
            "objective": "multiclass",
            "num_class": 3,
            "class_weight": "balanced",
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "subsample_for_bin": subsample_for_bin,
            "colsample_bytree": colsample_bytree,
            "subsample": subsample,
            "subsample_freq": subsample_freq,
            "min_child_samples": min_child_samples,
            "n_jobs": n_jobs,
            "random_state": random_state,
            "verbose": verbose,
            **kwargs,
        }
        self.classifier = lightgbm.LGBMClassifier(**self.params)

    def fit(
        self,
        games_features_list: list[np.ndarray],
        games_labels_list: list[np.ndarray],
    ) -> "PossessionLGBM":
        """Expand each continuous game sequence independently and fit LightGBM.

        Expanding sequences independently avoids boundary feature leakage across
        unrelated matches or augmented sequences.

        Args:
            games_features_list: List of (n_frames_i, 14) base feature arrays.
            games_labels_list: List of (n_frames_i,) ground truth integer labels.

        Returns:
            self.
        """
        expanded_list = [
            expand_temporal_tabular_features(feat) for feat in games_features_list
        ]
        x_all = np.concatenate(expanded_list, axis=0)
        y_all = np.concatenate(games_labels_list, axis=0)

        self.classifier.fit(x_all, y_all)
        return self

    def predict_proba(self, base_features: np.ndarray) -> np.ndarray:
        """Predict per-frame class probabilities for states (H, A, D).

        Args:
            base_features: Canonical 14-dimensional base feature matrix, or
                pre-expanded feature matrix.

        Returns:
            np.ndarray of shape (n_frames, 3), dtype float32.
        """
        if base_features.ndim != 2:
            raise ValueError(f"Expected 2D feature matrix, got shape {base_features.shape}")

        if base_features.shape[1] == 14:
            x_input = expand_temporal_tabular_features(base_features)
        else:
            x_input = base_features

        raw_probs = self.classifier.predict_proba(x_input)
        raw_probs = np.asarray(raw_probs, dtype=np.float32)

        if raw_probs.shape[1] == 3:
            return raw_probs

        classes = getattr(self.classifier, "classes_", np.array([0, 1, 2]))
        aligned_probs = np.zeros((len(base_features), 3), dtype=np.float32)
        for i, c in enumerate(classes):
            if 0 <= c < 3:
                aligned_probs[:, c] = raw_probs[:, i]
        return aligned_probs

    def save(self, path: str | Path) -> None:
        """Serialize model instance to disk using joblib."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, save_path)

    @classmethod
    def load(cls, path: str | Path) -> "PossessionLGBM":
        """Deserialize model instance from disk using joblib."""
        load_path = Path(path)
        if not load_path.is_file():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        loaded = joblib.load(load_path)
        if isinstance(loaded, cls):
            return loaded
        if isinstance(loaded, dict) and "model" in loaded:
            return loaded["model"]
        if hasattr(loaded, "predict_proba"):
            instance = cls()
            instance.classifier = loaded
            return instance
        raise TypeError(f"Loaded object from {load_path} is not a PossessionLGBM instance: {type(loaded)}")
