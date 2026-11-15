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

from dataclasses import dataclass

import numpy as np

# Canonical feature specifications
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
        np.ndarray of shape (n_frames, 14), dtype float32.
    """
    n_frames = len(is_ball_visible)
    features = np.zeros((n_frames, N_FEATURES), dtype=np.float32)

    length, width = pitch
    half_length = length / 2.0
    half_width = width / 2.0

    # Ensure boolean visibility mask
    visible_mask = np.asarray(is_ball_visible, dtype=bool) & ~np.isnan(ball_xy[:, 0])

    # 1. Collective outfield player speed statistics (features 9 and 10)
    if outfield_mean_speed is not None and outfield_p80_speed is not None:
        features[:, 9] = np.nan_to_num(outfield_mean_speed, nan=0.0).astype(np.float32)
        features[:, 10] = np.nan_to_num(outfield_p80_speed, nan=0.0).astype(np.float32)
    else:
        # Concatenate Home and Away outfield speeds
        combined_speeds = np.concatenate([home_player_speed, away_player_speed], axis=1)
        with np.errstate(invalid="ignore"):
            mean_speed = np.nanmean(combined_speeds, axis=1)
            p80_speed = np.nanpercentile(combined_speeds, 80, axis=1)
        features[:, 9] = np.nan_to_num(mean_speed, nan=0.0).astype(np.float32)
        features[:, 10] = np.nan_to_num(p80_speed, nan=0.0).astype(np.float32)

    # 2. Team outfield centroids normalized to [-1.0, 1.0] (features 11 and 12)
    if home_centroid_x_norm is not None:
        features[:, 11] = np.nan_to_num(home_centroid_x_norm, nan=0.0).astype(np.float32)
    else:
        with np.errstate(invalid="ignore"):
            h_mean_x = np.nanmean(home_player_xy[:, :, 0], axis=1)
        h_norm_x = (h_mean_x - half_length) / half_length
        features[:, 11] = np.clip(np.nan_to_num(h_norm_x, nan=0.0), -1.0, 1.0).astype(np.float32)

    if away_centroid_x_norm is not None:
        features[:, 12] = np.nan_to_num(away_centroid_x_norm, nan=0.0).astype(np.float32)
    else:
        with np.errstate(invalid="ignore"):
            a_mean_x = np.nanmean(away_player_xy[:, :, 0], axis=1)
        a_norm_x = (a_mean_x - half_length) / half_length
        features[:, 12] = np.clip(np.nan_to_num(a_norm_x, nan=0.0), -1.0, 1.0).astype(np.float32)

    # 3. Stoppage tag flag (feature 13)
    features[:, 13] = (np.asarray(is_stoppage_tag) > 0.5).astype(np.float32)

    # 4. Ball visibility indicator (feature 3)
    features[:, 3] = visible_mask.astype(np.float32)

    # When ball is not visible, distances default to sentinel and speeds/positions default to 0.0
    features[~visible_mask, 0] = MISSING_DIST_SENTINEL  # min_dist_home
    features[~visible_mask, 1] = MISSING_DIST_SENTINEL  # min_dist_away
    features[~visible_mask, 2] = 0.0                    # dist_diff
    features[~visible_mask, 4] = 0.0                    # ball_speed
    features[~visible_mask, 5] = 0.0                    # ball_x_norm
    features[~visible_mask, 6] = 0.0                    # ball_y_norm
    features[~visible_mask, 7] = 0.0                    # closest_home_speed
    features[~visible_mask, 8] = 0.0                    # closest_away_speed

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

        # Distances to Home outfield players
        vis_home_xy = home_player_xy[visible_indices]      # shape (V, n_home, 2)
        vis_home_spd = home_player_speed[visible_indices]  # shape (V, n_home)
        dx_h = vis_home_xy[:, :, 0] - vis_ball_xy[:, 0:1]
        dy_h = vis_home_xy[:, :, 1] - vis_ball_xy[:, 1:2]
        dist_h = np.sqrt(dx_h * dx_h + dy_h * dy_h)
        dist_h_nan = np.isnan(dist_h)
        dist_h[dist_h_nan] = np.inf

        min_idx_h = np.argmin(dist_h, axis=1)
        min_d_h = dist_h[np.arange(len(visible_indices)), min_idx_h]
        no_h = np.isinf(min_d_h)
        min_d_h[no_h] = MISSING_DIST_SENTINEL

        # Closest Home player speed
        spd_h = vis_home_spd[np.arange(len(visible_indices)), min_idx_h]
        spd_h[no_h | np.isnan(spd_h)] = 0.0

        # Distances to Away outfield players
        vis_away_xy = away_player_xy[visible_indices]      # shape (V, n_away, 2)
        vis_away_spd = away_player_speed[visible_indices]  # shape (V, n_away)
        dx_a = vis_away_xy[:, :, 0] - vis_ball_xy[:, 0:1]
        dy_a = vis_away_xy[:, :, 1] - vis_ball_xy[:, 1:2]
        dist_a = np.sqrt(dx_a * dx_a + dy_a * dy_a)
        dist_a_nan = np.isnan(dist_a)
        dist_a[dist_a_nan] = np.inf

        min_idx_a = np.argmin(dist_a, axis=1)
        min_d_a = dist_a[np.arange(len(visible_indices)), min_idx_a]
        no_a = np.isinf(min_d_a)
        min_d_a[no_a] = MISSING_DIST_SENTINEL

        # Closest Away player speed
        spd_a = vis_away_spd[np.arange(len(visible_indices)), min_idx_a]
        spd_a[no_a | np.isnan(spd_a)] = 0.0

        # Assign back
        features[visible_indices, 0] = min_d_h.astype(np.float32)
        features[visible_indices, 1] = min_d_a.astype(np.float32)
        features[visible_indices, 2] = (min_d_a - min_d_h).astype(np.float32)
        features[visible_indices, 7] = spd_h.astype(np.float32)
        features[visible_indices, 8] = spd_a.astype(np.float32)

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
