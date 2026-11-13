"""Ball trajectory reconstruction and ambiguity flagging for automatic tracking data.

Handles the ~41% missing ball observations in Metrica ATD by reconstructing short, physically
feasible occlusions (< 35 frames / < 1.4 seconds) and marking unresolved blackouts or erratic
tracking spikes as uncertain (for downstream abstention 'X' or state persistence).
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np
from scipy.interpolate import PchipInterpolator


@dataclass
class BallTrackingResult:
    """Result of ball trajectory reconstruction and uncertainty flagging."""
    ball_xy: np.ndarray           # shape (n_frames, 2), float32 in metres (reconstructed)
    is_ball_uncertain: np.ndarray # shape (n_frames,), bool (True where ball position is uncertain)
    ball_speed: np.ndarray        # shape (n_frames,), float32 in m/s (estimated speed)
    raw_missing_mask: np.ndarray  # shape (n_frames,), bool (True where raw observation was NaN)


def sanitize_raw_ball_positions(
    ball_xy: np.ndarray,
    pitch: Tuple[float, float],
    margin: float = 5.0,
    max_jump_speed_mps: float = 40.0,
    fps: float = 25.0,
) -> np.ndarray:
    """Filter out gross tracking anomalies and single-frame teleportation spikes.

    Args:
        ball_xy: Raw ball (x, y) coordinates in metres, shape (n_frames, 2).
        pitch: Pitch dimensions in metres (length, width).
        margin: Maximum allowable distance outside pitch boundaries before rejecting.
        max_jump_speed_mps: Maximum physically plausible speed between consecutive frames.
        fps: Video frame rate.

    Returns:
        Cleaned copy of ball_xy with invalid detections set to NaN.
    """
    clean_xy = ball_xy.copy()
    length, width = pitch
    n_frames = len(clean_xy)

    # 1. Pitch boundary sanity: coordinates far outside the stadium pitch are tracking artifacts
    out_of_bounds = (
        (clean_xy[:, 0] < -margin)
        | (clean_xy[:, 0] > length + margin)
        | (clean_xy[:, 1] < -margin)
        | (clean_xy[:, 1] > width + margin)
    )
    clean_xy[out_of_bounds] = np.nan

    # 2. Reject isolated single-frame teleportation spikes
    valid_indices = np.flatnonzero(~np.isnan(clean_xy[:, 0]))
    if len(valid_indices) < 2:
        return clean_xy

    dt = 1.0 / fps
    for idx_pos in range(len(valid_indices) - 1):
        i1 = valid_indices[idx_pos]
        i2 = valid_indices[idx_pos + 1]
        frame_gap = i2 - i1
        if frame_gap <= 2:
            time_gap = frame_gap * dt
            dx = clean_xy[i2, 0] - clean_xy[i1, 0]
            dy = clean_xy[i2, 1] - clean_xy[i1, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            implied_speed = dist / time_gap
            if implied_speed > max_jump_speed_mps:
                # Teleportation spike: invalidate the suspect observation if it quickly snaps back
                if idx_pos + 2 < len(valid_indices):
                    i3 = valid_indices[idx_pos + 2]
                    if (i3 - i2) <= 2:
                        dx_snap = clean_xy[i3, 0] - clean_xy[i1, 0]
                        dy_snap = clean_xy[i3, 1] - clean_xy[i1, 1]
                        dist_snap = np.sqrt(dx_snap * dx_snap + dy_snap * dy_snap)
                        speed_snap = dist_snap / ((i3 - i1) * dt)
                        if speed_snap <= max_jump_speed_mps:
                            clean_xy[i2] = np.nan

    return clean_xy


def find_missing_spans(is_missing: np.ndarray) -> List[Tuple[int, int]]:
    """Find contiguous spans [start_idx, end_idx) of missing observations.

    Args:
        is_missing: Boolean array where True indicates missing observation.

    Returns:
        List of (start, end) tuples, half-open on the right.
    """
    if not len(is_missing):
        return []

    diffs = np.diff(is_missing.astype(np.int8))
    starts = (np.flatnonzero(diffs == 1) + 1).tolist()
    ends = (np.flatnonzero(diffs == -1) + 1).tolist()

    if is_missing[0]:
        starts = [0] + starts
    if is_missing[-1]:
        ends = ends + [len(is_missing)]

    return list(zip(starts, ends))


def reconstruct_ball_trajectory(
    raw_ball_xy: np.ndarray,
    pitch: Tuple[float, float] = (105.3, 68.0),
    max_gap_frames: int = 35,
    max_physical_speed_mps: float = 38.0,
    fps: float = 25.0,
) -> BallTrackingResult:
    """Reconstruct missing ball positions across short occlusions and flag blackouts.

    Rules:
    - Missing gaps < max_gap_frames (1.4 s at 25 fps) with physically feasible endpoint
      velocities are smoothly interpolated.
    - Long blackouts (>= max_gap_frames) or gaps with physically impossible implied speeds
      (> 38 m/s) are NOT hallucinated; they are flagged as uncertain for downstream abstention ('X').

    Args:
        raw_ball_xy: Raw ball (x, y) coordinates in metres, shape (n_frames, 2).
        pitch: Pitch dimensions in metres (length, width).
        max_gap_frames: Maximum gap duration in frames to interpolate (default: 35 frames).
        max_physical_speed_mps: Maximum physically plausible average speed across a gap (default: 38.0 m/s).
        fps: Frames per second (default: 25.0).

    Returns:
        BallTrackingResult with reconstructed coordinates, uncertainty mask, and ball speed.
    """
    n_frames = len(raw_ball_xy)
    dt = 1.0 / fps

    # Sanitize raw coordinates
    cleaned_xy = sanitize_raw_ball_positions(
        raw_ball_xy,
        pitch=pitch,
        max_jump_speed_mps=max_physical_speed_mps + 5.0,
        fps=fps,
    )

    raw_missing = np.isnan(cleaned_xy[:, 0]) | np.isnan(cleaned_xy[:, 1])
    reconstructed_xy = cleaned_xy.copy()
    is_ball_uncertain = raw_missing.copy()

    # Find all contiguous missing spans
    spans = find_missing_spans(raw_missing)

    for start_idx, end_idx in spans:
        gap_len = end_idx - start_idx

        # Boundary gaps at beginning or end of recording cannot be safely interpolated
        if start_idx == 0 or end_idx >= n_frames:
            is_ball_uncertain[start_idx:end_idx] = True
            continue

        p_prev = reconstructed_xy[start_idx - 1]
        p_next = reconstructed_xy[end_idx]

        # Verify endpoint validity
        if np.isnan(p_prev[0]) or np.isnan(p_next[0]):
            is_ball_uncertain[start_idx:end_idx] = True
            continue

        time_gap = (gap_len + 1) * dt
        dx = p_next[0] - p_prev[0]
        dy = p_next[1] - p_prev[1]
        dist = np.sqrt(dx * dx + dy * dy)
        avg_speed = dist / time_gap

        # Only interpolate if gap is short AND implied speed is physically realistic
        if gap_len < max_gap_frames and avg_speed <= max_physical_speed_mps:
            # Interpolate smoothly across the short gap
            # For short spans (<= 4 frames), linear interpolation is optimal and overshoot-free
            # For medium spans (5 to 34 frames), use shape-preserving PCHIP if enough context exists
            gap_frame_indices = np.arange(start_idx, end_idx)

            if gap_len <= 4:
                # Fast linear interpolation
                alphas = (np.arange(1, gap_len + 1) / (gap_len + 1)).reshape(-1, 1)
                interp_points = p_prev + alphas * (p_next - p_prev)
                reconstructed_xy[start_idx:end_idx] = interp_points
                is_ball_uncertain[start_idx:end_idx] = False
            else:
                # Use local window of valid points for smooth, non-oscillating PCHIP
                left_ctx = max(0, start_idx - 10)
                right_ctx = min(n_frames, end_idx + 10)

                valid_left = [i for i in range(left_ctx, start_idx) if not np.isnan(reconstructed_xy[i, 0])]
                valid_right = [i for i in range(end_idx, right_ctx) if not np.isnan(reconstructed_xy[i, 0])]

                if len(valid_left) >= 2 and len(valid_right) >= 2:
                    ctx_indices = np.array(valid_left[-3:] + valid_right[:3])
                    ctx_x = reconstructed_xy[ctx_indices, 0]
                    ctx_y = reconstructed_xy[ctx_indices, 1]

                    pchip_x = PchipInterpolator(ctx_indices, ctx_x)
                    pchip_y = PchipInterpolator(ctx_indices, ctx_y)

                    reconstructed_xy[start_idx:end_idx, 0] = pchip_x(gap_frame_indices)
                    reconstructed_xy[start_idx:end_idx, 1] = pchip_y(gap_frame_indices)
                else:
                    # Fallback to linear interpolation
                    alphas = (np.arange(1, gap_len + 1) / (gap_len + 1)).reshape(-1, 1)
                    reconstructed_xy[start_idx:end_idx] = p_prev + alphas * (p_next - p_prev)

                is_ball_uncertain[start_idx:end_idx] = False
        else:
            # Long blackout or erratic jump: do NOT hallucinate
            is_ball_uncertain[start_idx:end_idx] = True

    # Compute ball speed via central differences on valid coordinates
    ball_speed = np.full(n_frames, np.nan, dtype=np.float32)
    valid_coords = ~is_ball_uncertain

    for i in range(n_frames):
        if not valid_coords[i]:
            continue
        has_prev = (i > 0) and valid_coords[i - 1]
        has_next = (i < n_frames - 1) and valid_coords[i + 1]

        if has_prev and has_next:
            vx = (reconstructed_xy[i + 1, 0] - reconstructed_xy[i - 1, 0]) / (2.0 * dt)
            vy = (reconstructed_xy[i + 1, 1] - reconstructed_xy[i - 1, 1]) / (2.0 * dt)
            ball_speed[i] = np.sqrt(vx * vx + vy * vy)
        elif has_next:
            vx = (reconstructed_xy[i + 1, 0] - reconstructed_xy[i, 0]) / dt
            vy = (reconstructed_xy[i + 1, 1] - reconstructed_xy[i, 1]) / dt
            ball_speed[i] = np.sqrt(vx * vx + vy * vy)
        elif has_prev:
            vx = (reconstructed_xy[i, 0] - reconstructed_xy[i - 1, 0]) / dt
            vy = (reconstructed_xy[i, 1] - reconstructed_xy[i - 1, 1]) / dt
            ball_speed[i] = np.sqrt(vx * vx + vy * vy)
        else:
            ball_speed[i] = 0.0

    return BallTrackingResult(
        ball_xy=reconstructed_xy.astype(np.float32),
        is_ball_uncertain=is_ball_uncertain,
        ball_speed=ball_speed,
        raw_missing_mask=raw_missing,
    )
