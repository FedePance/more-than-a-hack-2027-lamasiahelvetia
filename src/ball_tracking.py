"""Ball trajectory reconstruction and ambiguity flagging for automatic tracking data.

Handles the ~41% missing ball observations in Metrica ATD by:
1. Bidirectional kinematic reachability filtering to eliminate 1-3 frame false-positive teleportation spikes.
2. Savitzky-Golay trajectory smoothing on contiguous raw observed segments before interpolation.
3. Central-difference velocity estimation on smoothed raw segments.
4. Monotonic shape-preserving interpolation (PchipInterpolator) with exact analytic derivatives across short occlusions (< 35 frames).
5. Marking prolonged blackouts (>= 35 frames) as uncertain for downstream abstention ('X').
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter

from src.mask_utils import find_true_runs

# Number of valid neighbours inspected / used on each side of a gap for PCHIP interpolation
_GAP_CONTEXT_SEARCH_FRAMES = 10
_GAP_CONTEXT_POINTS = 3
_MAX_SPIKE_LENGTH = 3


@dataclass
class BallTrackingResult:
    """Result of ball trajectory reconstruction and uncertainty flagging."""
    ball_xy: np.ndarray           # shape (n_frames, 2), float32 in metres (smoothed & reconstructed)
    is_ball_uncertain: np.ndarray # shape (n_frames,), bool (True where ball position is uncertain)
    ball_speed: np.ndarray        # shape (n_frames,), float32 in m/s (analytic & central-difference speed)
    raw_missing_mask: np.ndarray  # shape (n_frames,), bool (True where raw observation was NaN)


def _spike_candidate_positions(
    points: np.ndarray,
    frames: np.ndarray,
    dt: float,
    max_jump_speed_mps: float,
) -> np.ndarray:
    """Return positions (into the valid-detection list) where a spike can possibly start.

    A spike starting right after valid detection i requires detection i + 1 to be unreachable
    either from detection i or from one of the next three detections. Positions failing this
    necessary condition can be skipped without running the full reachability test.
    """
    n_valid = len(points)
    candidate = np.zeros(n_valid, dtype=bool)

    def pair_speed_exceeds(lag: int) -> np.ndarray:
        delta = points[lag:] - points[:-lag]
        dist = np.sqrt(np.sum(delta ** 2, axis=1))
        speed = dist / ((frames[lag:] - frames[:-lag]) * dt)
        # NaN-safe: an undefined speed counts as unreachable, as in the scalar test
        return ~(speed <= max_jump_speed_mps)

    # Detection i + 1 unreachable from detection i
    candidate[: n_valid - 1] |= pair_speed_exceeds(1)
    # Detection i + 1 unreachable from detection i + 1 + lag
    for lag in range(1, _MAX_SPIKE_LENGTH + 1):
        if n_valid - lag - 1 > 0:
            candidate[: n_valid - lag - 1] |= pair_speed_exceeds(lag)[1:]

    return np.flatnonzero(candidate)


def sanitize_raw_ball_positions(
    ball_xy: np.ndarray,
    pitch: Tuple[float, float],
    margin: float = 5.0,
    max_jump_speed_mps: float = 40.0,
    fps: float = 25.0,
) -> np.ndarray:
    """Filter out stadium boundary anomalies and 1-3 frame false-positive tracking spikes.

    Uses a bidirectional kinematic reachability check: if a sequence of 1 to 3 intermediate
    detections requires physically impossible speeds (> 40 m/s) to reach from the preceding
    anchor point and to return to the subsequent anchor point, while the two anchor points are
    themselves mutually reachable, the intermediate detections are flagged as false-positive spikes.

    Args:
        ball_xy: Raw ball (x, y) coordinates in metres, shape (n_frames, 2).
        pitch: Pitch dimensions in metres (length, width).
        margin: Maximum allowable distance outside pitch boundaries before rejecting.
        max_jump_speed_mps: Maximum physically plausible speed between frames (~144 km/h).
        fps: Video frame rate.

    Returns:
        Cleaned copy of ball_xy with invalid detections set to NaN.
    """
    clean_xy = ball_xy.copy()
    length, width = pitch
    dt = 1.0 / fps

    # 0. A detection is only usable when both coordinates are finite
    clean_xy[~np.isfinite(clean_xy).all(axis=1)] = np.nan

    # 1. Pitch boundary sanity check
    out_of_bounds = (
        (clean_xy[:, 0] < -margin)
        | (clean_xy[:, 0] > length + margin)
        | (clean_xy[:, 1] < -margin)
        | (clean_xy[:, 1] > width + margin)
    )
    clean_xy[out_of_bounds] = np.nan

    # 2. Bidirectional kinematic reachability check for 1-3 frame false-positive spikes
    valid_indices = np.flatnonzero(~np.isnan(clean_xy[:, 0]))
    n_valid = len(valid_indices)
    if n_valid < 3:
        return clean_xy

    # Vectorized pre-filter: the scalar test below only has to run where a spike is possible
    candidate_positions = _spike_candidate_positions(
        clean_xy[valid_indices], valid_indices, dt, max_jump_speed_mps
    )

    idx_pos = 0
    while True:
        next_candidate = int(np.searchsorted(candidate_positions, idx_pos))
        if next_candidate >= len(candidate_positions):
            break
        idx_pos = int(candidate_positions[next_candidate])
        if idx_pos >= n_valid - 1:
            break

        i_curr = valid_indices[idx_pos]
        p_curr = clean_xy[i_curr]

        # Test spike lengths k in {1, 2, 3} frames
        spike_found = False
        for k in range(1, _MAX_SPIKE_LENGTH + 1):
            anchor_pos = idx_pos + k + 1
            if anchor_pos >= n_valid:
                break

            i_anchor = valid_indices[anchor_pos]
            p_anchor = clean_xy[i_anchor]

            # Direct speed between current point and anchor point
            direct_time = (i_anchor - i_curr) * dt
            direct_dist = np.sqrt(np.sum((p_anchor - p_curr) ** 2))
            direct_speed = direct_dist / direct_time

            # Only consider the spike hypothesis when the anchor is directly reachable
            if not direct_speed <= max_jump_speed_mps:
                continue

            # Check if all intermediate points require impossible speed
            intermediates = valid_indices[idx_pos + 1 : anchor_pos]
            all_intermediates_impossible = True

            for i_inter in intermediates:
                p_inter = clean_xy[i_inter]
                speed_from_curr = np.sqrt(np.sum((p_inter - p_curr) ** 2)) / ((i_inter - i_curr) * dt)
                speed_to_anchor = np.sqrt(np.sum((p_anchor - p_inter) ** 2)) / ((i_anchor - i_inter) * dt)

                if speed_from_curr <= max_jump_speed_mps and speed_to_anchor <= max_jump_speed_mps:
                    all_intermediates_impossible = False
                    break

            if all_intermediates_impossible:
                # Invalidate intermediate spike detections and advance directly to the anchor
                clean_xy[intermediates] = np.nan
                idx_pos = anchor_pos
                spike_found = True
                break

        if not spike_found:
            idx_pos += 1

    return clean_xy


def find_missing_spans(is_missing: np.ndarray) -> List[Tuple[int, int]]:
    """Find contiguous spans [start_idx, end_idx) of missing observations."""
    starts, ends = find_true_runs(is_missing)
    return list(zip(starts.tolist(), ends.tolist()))


def _central_difference_speed(xy_block: np.ndarray, dt: float) -> np.ndarray:
    """Speed (m/s) along one contiguous observed block, shape (block_len,), float32.

    Interior frames use central differences; the two block edges use one-sided differences.
    A single-frame block has no neighbour to difference against and gets speed 0.0.
    """
    block_len = len(xy_block)
    if block_len < 2:
        return np.zeros(block_len, dtype=np.float32)

    velocity = np.empty((block_len, 2), dtype=np.float64)
    velocity[1:-1] = (xy_block[2:] - xy_block[:-2]).astype(np.float64) / (2.0 * dt)
    velocity[0] = (xy_block[1] - xy_block[0]).astype(np.float64) / dt
    velocity[-1] = (xy_block[-1] - xy_block[-2]).astype(np.float64) / dt
    return np.sqrt(velocity[:, 0] ** 2 + velocity[:, 1] ** 2).astype(np.float32)


def reconstruct_ball_trajectory(
    raw_ball_xy: np.ndarray,
    pitch: Tuple[float, float] = (105.3, 68.0),
    max_gap_frames: int = 35,
    max_physical_speed_mps: float = 38.0,
    fps: float = 25.0,
    savgol_window: int = 7,
    savgol_polyorder: int = 2,
) -> BallTrackingResult:
    """Reconstruct missing ball positions across short occlusions and flag blackouts.

    Rules:
    - Continuous raw valid ball position segments are smoothed with a Savitzky-Golay filter
      before interpolation to eliminate quantization jitter.
    - Central-difference velocities are computed on the smoothed raw segments.
    - Missing gaps < max_gap_frames (1.4 s at 25 fps) with physically feasible endpoint velocities
      are smoothly interpolated using monotonic shape-preserving PchipInterpolator.
    - Inside interpolated gaps, ball velocities are computed using exact analytic PCHIP derivatives,
      ensuring 100% kinematic consistency across gap boundaries.
    - Long blackouts (>= max_gap_frames) or gaps with physically impossible implied speeds (> 38 m/s)
      are NOT hallucinated; they are flagged as uncertain for downstream abstention ('X').

    Args:
        raw_ball_xy: Raw ball (x, y) coordinates in metres, shape (n_frames, 2).
        pitch: Pitch dimensions in metres (length, width).
        max_gap_frames: Maximum gap duration in frames to interpolate (default: 35 frames).
        max_physical_speed_mps: Maximum physically plausible average speed across a gap (default: 38.0 m/s).
        fps: Frames per second (default: 25.0).
        savgol_window: Window length for Savitzky-Golay smoothing (default: 7).
        savgol_polyorder: Polynomial order for Savitzky-Golay smoothing (default: 2).

    Returns:
        BallTrackingResult with reconstructed coordinates, uncertainty mask, and ball speed.
        Frames flagged uncertain keep NaN coordinates and NaN speed.

    Raises:
        ValueError: If the input shape or the smoothing / timing parameters are invalid.
    """
    raw_ball_xy = np.asarray(raw_ball_xy)
    if raw_ball_xy.ndim != 2 or raw_ball_xy.shape[1] != 2:
        raise ValueError(f"raw_ball_xy must have shape (n_frames, 2), got {raw_ball_xy.shape}")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if savgol_window % 2 == 0 or savgol_window <= savgol_polyorder:
        raise ValueError(
            f"savgol_window must be odd and greater than savgol_polyorder, "
            f"got window={savgol_window}, polyorder={savgol_polyorder}"
        )

    n_frames = len(raw_ball_xy)
    dt = 1.0 / fps

    # Sanitize raw coordinates with bidirectional reachability
    cleaned_xy = sanitize_raw_ball_positions(
        raw_ball_xy,
        pitch=pitch,
        max_jump_speed_mps=max_physical_speed_mps + 2.0,
        fps=fps,
    )

    raw_missing = np.isnan(cleaned_xy).any(axis=1)
    reconstructed_xy = cleaned_xy.copy()
    is_ball_uncertain = raw_missing.copy()
    ball_speed = np.full(n_frames, np.nan, dtype=np.float32)

    # 1. Smooth each contiguous observed block and compute central-difference speeds
    #    BEFORE interpolating missing gaps
    observed_starts, observed_ends = find_true_runs(~raw_missing)
    for s, e in zip(observed_starts.tolist(), observed_ends.tolist()):
        if e - s >= savgol_window:
            for axis in (0, 1):
                reconstructed_xy[s:e, axis] = savgol_filter(
                    reconstructed_xy[s:e, axis], window_length=savgol_window, polyorder=savgol_polyorder
                )
        ball_speed[s:e] = _central_difference_speed(reconstructed_xy[s:e], dt)

    # 2. Interpolate missing gaps using PchipInterpolator and exact analytic derivatives
    for start_idx, end_idx in find_missing_spans(raw_missing):
        gap_len = end_idx - start_idx

        # Boundary gaps at beginning or end of recording cannot be safely interpolated
        if start_idx == 0 or end_idx >= n_frames:
            continue

        p_prev = reconstructed_xy[start_idx - 1]
        p_next = reconstructed_xy[end_idx]

        time_gap = (gap_len + 1) * dt
        dx = p_next[0] - p_prev[0]
        dy = p_next[1] - p_prev[1]
        avg_speed = np.sqrt(dx * dx + dy * dy) / time_gap

        # Long blackouts and physically impossible jumps stay uncertain (NaN-safe comparison)
        if gap_len >= max_gap_frames or not avg_speed <= max_physical_speed_mps:
            continue

        gap_frame_indices = np.arange(start_idx, end_idx)

        left_ctx = max(0, start_idx - _GAP_CONTEXT_SEARCH_FRAMES)
        right_ctx = min(n_frames, end_idx + _GAP_CONTEXT_SEARCH_FRAMES)
        valid_left = left_ctx + np.flatnonzero(~np.isnan(reconstructed_xy[left_ctx:start_idx, 0]))
        valid_right = end_idx + np.flatnonzero(~np.isnan(reconstructed_xy[end_idx:right_ctx, 0]))

        if len(valid_left) >= 2 and len(valid_right) >= 2:
            ctx_indices = np.concatenate(
                [valid_left[-_GAP_CONTEXT_POINTS:], valid_right[:_GAP_CONTEXT_POINTS]]
            )
            pchip_x = PchipInterpolator(ctx_indices, reconstructed_xy[ctx_indices, 0])
            pchip_y = PchipInterpolator(ctx_indices, reconstructed_xy[ctx_indices, 1])

            reconstructed_xy[start_idx:end_idx, 0] = pchip_x(gap_frame_indices)
            reconstructed_xy[start_idx:end_idx, 1] = pchip_y(gap_frame_indices)

            # Analytic velocity derivatives (d/d_frame * fps = d/d_second)
            vx_analytic = pchip_x.derivative(1)(gap_frame_indices) * fps
            vy_analytic = pchip_y.derivative(1)(gap_frame_indices) * fps
            ball_speed[start_idx:end_idx] = np.sqrt(vx_analytic * vx_analytic + vy_analytic * vy_analytic)
        else:
            alphas = (np.arange(1, gap_len + 1) / (gap_len + 1)).reshape(-1, 1)
            reconstructed_xy[start_idx:end_idx] = p_prev + alphas * (p_next - p_prev)
            vx_lin = dx / time_gap
            vy_lin = dy / time_gap
            ball_speed[start_idx:end_idx] = float(np.sqrt(vx_lin * vx_lin + vy_lin * vy_lin))

        is_ball_uncertain[start_idx:end_idx] = False

    return BallTrackingResult(
        ball_xy=reconstructed_xy.astype(np.float32),
        is_ball_uncertain=is_ball_uncertain,
        ball_speed=ball_speed,
        raw_missing_mask=raw_missing,
    )
