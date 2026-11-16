"""Unified domain-invariant feature extractor for ball possession identification.

Produces 100% identical 14-dimensional feature representations for both open
Metrica data (Sample Games 1-3) and competition tracking data (BAR-ATH).

Features (14 dimensions):
    0. min_dist_home: Distance (m) from ball to closest outfield Home player (sentinel 50.0 if ball missing).
    1. min_dist_away: Distance (m) from ball to closest outfield Away player (sentinel 50.0 if ball missing).
    2. dist_diff: min_dist_away - min_dist_home (0.0 if ball missing).
    3. is_ball_visible: 1.0 if ball is tracked/valid, 0.0 if missing/occluded.
    4. ball_speed: Ball speed in m/s (0.0 if ball missing).
    5. ball_x_norm: Longitudinal ball position in [-1.0, 1.0] relative to pitch center (0.0 if missing).
    6. ball_y_norm: Lateral ball position in [-1.0, 1.0] relative to pitch center (0.0 if missing).
    7. closest_home_speed: Speed (m/s) of the nearest Home outfield player to the ball (0.0 if missing).
    8. closest_away_speed: Speed (m/s) of the nearest Away outfield player to the ball (0.0 if missing).
    9. outfield_mean_speed: Mean speed (m/s) of all active outfield players.
    10. outfield_p80_speed: 80th-percentile speed (m/s) of all active outfield players.
    11. home_centroid_x_norm: Mean longitudinal position of Home outfield players in [-1.0, 1.0].
    12. away_centroid_x_norm: Mean longitudinal position of Away outfield players in [-1.0, 1.0].
    13. is_stoppage_tag: 1.0 during stoppage / set-piece setup intervals, 0.0 otherwise.
"""

import warnings
from dataclasses import dataclass

import numpy as np

# Canonical feature specifications (order and semantics are frozen: trained checkpoints depend on them)
FEATURE_NAMES = [
    "min_dist_home",
    "min_dist_away",
    "dist_diff",
    "is_ball_visible",
    "ball_speed",
    "ball_x_norm",
    "ball_y_norm",
    "closest_home_speed",
    "closest_away_speed",
    "outfield_mean_speed",
    "outfield_p80_speed",
    "home_centroid_x_norm",
    "away_centroid_x_norm",
    "is_stoppage_tag",
]
N_FEATURES = len(FEATURE_NAMES)
MISSING_DIST_SENTINEL = 50.0
BALL_VISIBLE_INDEX = FEATURE_NAMES.index("is_ball_visible")


def mask_ball_features(features: np.ndarray, masked: np.ndarray) -> np.ndarray:
    """Reset every ball-dependent channel (features 0-8) on frames where the ball is unobserved.

    Distances fall back to MISSING_DIST_SENTINEL; visibility, ball speed, ball position and
    closest-player speeds fall back to 0.0. This is the single definition of what a
    "ball missing" frame looks like, shared by feature extraction, augmentation and validation.

    Args:
        features: Feature matrix of shape (n_frames, 14), modified in place.
        masked: Boolean mask of shape (n_frames,), True where the ball is missing / occluded.

    Returns:
        The same features array, for chaining.
    """
    features[masked, 0:2] = MISSING_DIST_SENTINEL  # min_dist_home, min_dist_away
    features[masked, 2:9] = 0.0                    # dist_diff .. closest_away_speed
    return features


def _closest_player_stats(
    player_xy: np.ndarray,
    player_speed: np.ndarray,
    ball_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Distance to, and speed of, the player closest to the ball on each frame.

    Args:
        player_xy: Positions of one team's outfield players, shape (n, n_players, 2).
        player_speed: Their speeds, shape (n, n_players).
        ball_xy: Ball positions, shape (n, 2).

    Returns:
        Tuple (min_dist, closest_speed), each of shape (n,). Frames with no observed player
        get MISSING_DIST_SENTINEL and speed 0.0.
    """
    n_rows, n_players = player_xy.shape[:2]
    if n_players == 0:
        return (
            np.full(n_rows, MISSING_DIST_SENTINEL, dtype=np.float32),
            np.zeros(n_rows, dtype=np.float32),
        )

    dx = player_xy[:, :, 0] - ball_xy[:, 0:1]
    dy = player_xy[:, :, 1] - ball_xy[:, 1:2]
    dist = np.sqrt(dx * dx + dy * dy)
    dist[np.isnan(dist)] = np.inf

    rows = np.arange(n_rows)
    min_idx = np.argmin(dist, axis=1)
    min_dist = dist[rows, min_idx]
    no_player = np.isinf(min_dist)
    min_dist[no_player] = MISSING_DIST_SENTINEL

    closest_speed = player_speed[rows, min_idx]
    closest_speed[no_player | np.isnan(closest_speed)] = 0.0
    return min_dist, closest_speed


@dataclass
class MatchFrameInputs:
    """Structured container holding per-frame match tracking observations."""

    ball_xy: np.ndarray  # shape (n_frames, 2), in metres
    ball_speed: np.ndarray  # shape (n_frames,), in m/s
    is_ball_visible: np.ndarray  # shape (n_frames,), bool
    home_player_xy: np.ndarray  # shape (n_frames, n_home, 2), in metres
    home_player_speed: np.ndarray  # shape (n_frames, n_home), in m/s
    away_player_xy: np.ndarray  # shape (n_frames, n_away, 2), in metres
    away_player_speed: np.ndarray  # shape (n_frames, n_away), in m/s
    is_stoppage_tag: np.ndarray  # shape (n_frames,), bool or float32
    pitch: tuple[float, float] = (105.3, 68.0)
    outfield_mean_speed: np.ndarray | None = None
    outfield_p80_speed: np.ndarray | None = None
    home_centroid_x_norm: np.ndarray | None = None
    away_centroid_x_norm: np.ndarray | None = None


def extract_frame_features(
    ball_xy: np.ndarray,
    ball_speed: np.ndarray,
    is_ball_visible: np.ndarray,
    home_player_xy: np.ndarray,
    home_player_speed: np.ndarray,
    away_player_xy: np.ndarray,
    away_player_speed: np.ndarray,
    is_stoppage_tag: np.ndarray,
    pitch: tuple[float, float] = (105.3, 68.0),
    outfield_mean_speed: np.ndarray | None = None,
    outfield_p80_speed: np.ndarray | None = None,
    home_centroid_x_norm: np.ndarray | None = None,
    away_centroid_x_norm: np.ndarray | None = None,
) -> np.ndarray:
    """Extract canonical 14-dimensional feature matrix across all frames.

    Args:
        ball_xy: Reconstructed/raw ball positions, shape (n_frames, 2), in metres.
        ball_speed: Ball speed series, shape (n_frames,), in m/s.
        is_ball_visible: Boolean array indicating whether ball is tracked/valid, shape (n_frames,).
        home_player_xy: Home outfield player positions, shape (n_frames, n_home, 2), in metres.
        home_player_speed: Home outfield player speeds, shape (n_frames, n_home), in m/s.
        away_player_xy: Away outfield player positions, shape (n_frames, n_away, 2), in metres.
        away_player_speed: Away outfield player speeds, shape (n_frames, n_away), in m/s.
        is_stoppage_tag: Boolean/float mask indicating stoppage tag intervals, shape (n_frames,).
        pitch: Pitch dimensions in metres (length, width). Default: (105.3, 68.0).
        outfield_mean_speed: Optional precomputed collective outfield mean speed, shape (n_frames,).
        outfield_p80_speed: Optional precomputed collective outfield 80th percentile speed, shape (n_frames,).
        home_centroid_x_norm: Optional precomputed Home outfield normalized x centroid, shape (n_frames,).
        away_centroid_x_norm: Optional precomputed Away outfield normalized x centroid, shape (n_frames,).

    Returns:
        np.ndarray of shape (n_frames, 14), dtype float32, guaranteed finite.

    Raises:
        ValueError: If a per-frame input does not have the same number of rows as is_ball_visible.
    """
    n_frames = len(is_ball_visible)
    for name, values in (
        ("ball_xy", ball_xy),
        ("ball_speed", ball_speed),
        ("home_player_xy", home_player_xy),
        ("home_player_speed", home_player_speed),
        ("away_player_xy", away_player_xy),
        ("away_player_speed", away_player_speed),
        ("is_stoppage_tag", is_stoppage_tag),
    ):
        if len(values) != n_frames:
            raise ValueError(f"{name} has {len(values)} rows, expected {n_frames}")

    features = np.zeros((n_frames, N_FEATURES), dtype=np.float32)

    length, width = pitch
    half_length = length / 2.0
    half_width = width / 2.0

    # Ensure boolean visibility mask
    visible_mask = np.asarray(is_ball_visible, dtype=bool) & ~np.isnan(ball_xy[:, 0])

    def centroid_x_norm(player_xy: np.ndarray) -> np.ndarray:
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # frames with no tracked player
            mean_x = np.nanmean(player_xy[:, :, 0], axis=1)
        norm_x = (mean_x - half_length) / half_length
        return np.clip(np.nan_to_num(norm_x, nan=0.0), -1.0, 1.0).astype(np.float32)

    # 1. Collective outfield player speed statistics (features 9 and 10)
    if outfield_mean_speed is not None and outfield_p80_speed is not None:
        features[:, 9] = np.nan_to_num(outfield_mean_speed, nan=0.0).astype(np.float32)
        features[:, 10] = np.nan_to_num(outfield_p80_speed, nan=0.0).astype(np.float32)
    else:
        # Concatenate Home and Away outfield speeds
        combined_speeds = np.concatenate([home_player_speed, away_player_speed], axis=1)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # frames with no tracked player
            mean_speed = np.nanmean(combined_speeds, axis=1)
            p80_speed = np.nanpercentile(combined_speeds, 80, axis=1)
        features[:, 9] = np.nan_to_num(mean_speed, nan=0.0).astype(np.float32)
        features[:, 10] = np.nan_to_num(p80_speed, nan=0.0).astype(np.float32)

    # 2. Team outfield centroids normalized to [-1.0, 1.0] (features 11 and 12)
    if home_centroid_x_norm is not None:
        features[:, 11] = np.nan_to_num(home_centroid_x_norm, nan=0.0).astype(np.float32)
    else:
        features[:, 11] = centroid_x_norm(home_player_xy)

    if away_centroid_x_norm is not None:
        features[:, 12] = np.nan_to_num(away_centroid_x_norm, nan=0.0).astype(np.float32)
    else:
        features[:, 12] = centroid_x_norm(away_player_xy)

    # 3. Stoppage tag flag (feature 13)
    features[:, 13] = (np.asarray(is_stoppage_tag) > 0.5).astype(np.float32)

    # 4. Ball visibility indicator (feature 3)
    features[:, BALL_VISIBLE_INDEX] = visible_mask.astype(np.float32)

    # When ball is not visible, distances default to sentinel and speeds/positions default to 0.0
    mask_ball_features(features, ~visible_mask)

    # For frames where ball is visible, compute exact spatial relationships
    visible_indices = np.flatnonzero(visible_mask)
    if len(visible_indices) > 0:
        vis_ball_xy = ball_xy[visible_indices]  # shape (V, 2)
        vis_ball_speed = ball_speed[visible_indices]

        # Ball speed and normalized position
        features[visible_indices, 4] = np.nan_to_num(vis_ball_speed, nan=0.0).astype(np.float32)
        norm_bx = (vis_ball_xy[:, 0] - half_length) / half_length
        norm_by = (vis_ball_xy[:, 1] - half_width) / half_width
        features[visible_indices, 5] = np.clip(np.nan_to_num(norm_bx, nan=0.0), -1.0, 1.0).astype(np.float32)
        features[visible_indices, 6] = np.clip(np.nan_to_num(norm_by, nan=0.0), -1.0, 1.0).astype(np.float32)

        # Distance to, and speed of, the closest outfield player of each team
        min_d_h, spd_h = _closest_player_stats(
            home_player_xy[visible_indices], home_player_speed[visible_indices], vis_ball_xy
        )
        min_d_a, spd_a = _closest_player_stats(
            away_player_xy[visible_indices], away_player_speed[visible_indices], vis_ball_xy
        )

        features[visible_indices, 0] = min_d_h.astype(np.float32)
        features[visible_indices, 1] = min_d_a.astype(np.float32)
        features[visible_indices, 2] = (min_d_a - min_d_h).astype(np.float32)
        features[visible_indices, 7] = spd_h.astype(np.float32)
        features[visible_indices, 8] = spd_a.astype(np.float32)

    # Final safeguard: a non-finite value must never reach a model (no-op on clean inputs)
    if not np.isfinite(features).all():
        np.nan_to_num(features, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    return features


def extract_frame_features_from_tracking(
    tracking_data,
    is_stoppage_tag: np.ndarray,
    ball_tracking_res=None,
) -> np.ndarray:
    """Convenience bridge to extract features from RawTrackingData and optional BallTrackingResult.

    Args:
        tracking_data: RawTrackingData container from src/data_loader.py.
        is_stoppage_tag: Boolean/float mask indicating stoppage tag intervals, shape (n_frames,).
        ball_tracking_res: Optional BallTrackingResult container from src/ball_tracking.py.

    Returns:
        np.ndarray of shape (n_frames, 14), dtype float32.
    """
    pitch = tracking_data.pitch

    # Separate outfield slots by team
    home_slots = np.flatnonzero((tracking_data.team_of_track == 0) & tracking_data.is_outfield_slot)
    away_slots = np.flatnonzero((tracking_data.team_of_track == 1) & tracking_data.is_outfield_slot)

    home_player_xy = tracking_data.player_xy[:, home_slots]
    home_player_speed = tracking_data.player_speed[:, home_slots]
    away_player_xy = tracking_data.player_xy[:, away_slots]
    away_player_speed = tracking_data.player_speed[:, away_slots]

    if ball_tracking_res is not None:
        ball_xy = ball_tracking_res.ball_xy
        ball_speed = ball_tracking_res.ball_speed
        is_ball_visible = ~ball_tracking_res.is_ball_uncertain & ~np.isnan(ball_xy[:, 0])
    else:
        ball_xy = tracking_data.ball_xy
        ball_speed = tracking_data.ball_speed
        is_ball_visible = ~np.isnan(ball_xy[:, 0])

    return extract_frame_features(
        ball_xy=ball_xy,
        ball_speed=ball_speed,
        is_ball_visible=is_ball_visible,
        home_player_xy=home_player_xy,
        home_player_speed=home_player_speed,
        away_player_xy=away_player_xy,
        away_player_speed=away_player_speed,
        is_stoppage_tag=is_stoppage_tag,
        pitch=pitch,
        outfield_mean_speed=tracking_data.mean_player_speed,
        outfield_p80_speed=tracking_data.p80_player_speed,
    )
