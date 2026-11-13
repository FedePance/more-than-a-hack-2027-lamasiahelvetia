"""Ball trajectory reconstruction and ambiguity flagging for automatic tracking data.

Handles the ~41% missing ball observations in Metrica ATD by:
1. Bidirectional kinematic reachability filtering to eliminate 1-3 frame false-positive teleportation spikes.
2. Monotonic shape-preserving interpolation (PchipInterpolator) for short occlusions (< 35 frames).
3. Savitzky-Golay trajectory smoothing and exact analytic PCHIP derivatives for jitter-free ball velocities.
4. Marking prolonged blackouts (>= 35 frames) as uncertain for downstream abstention ('X').
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter


@dataclass
class BallTrackingResult:
    """Result of ball trajectory reconstruction and uncertainty flagging."""
    ball_xy: np.ndarray           # shape (n_frames, 2), float32 in metres (smoothed & reconstructed)
    is_ball_uncertain: np.ndarray # shape (n_frames,), bool (True where ball position is uncertain)
    ball_speed: np.ndarray        # shape (n_frames,), float32 in m/s (analytic & central-difference speed)
    raw_missing_mask: np.ndarray  # shape (n_frames,), bool (True where raw observation was NaN)


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
    n_frames = len(clean_xy)
    dt = 1.0 / fps

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
    if len(valid_indices) < 3:
        return clean_xy

    idx_pos = 0
    n_valid = len(valid_indices)

    while idx_pos < n_valid - 1:
        i_curr = valid_indices[idx_pos]
        p_curr = clean_xy[i_curr]

        # Test spike lengths k in {1, 2, 3} frames
        spike_found = False
        for k in (1, 2, 3):
            anchor_pos = idx_pos + k + 1
            if anchor_pos >= n_valid:
                break

            i_anchor = valid_indices[anchor_pos]
            p_anchor = clean_xy[i_anchor]

            # Direct speed between current point and anchor point
            direct_time = (i_anchor - i_curr) * dt
            direct_dist = np.sqrt(np.sum((p_anchor - p_curr) ** 2))
            direct_speed = direct_dist / direct_time

            # If anchor is directly reachable
            if direct_speed <= max_jump_speed_mps:
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

                if all_intermediates_impossible and len(intermediates) > 0:
                    # Invalidate intermediate spike detections
                    for i_inter in intermediates:
                        clean_xy[i_inter] = np.nan
                    # Advance directly to the anchor point
                    idx_pos = anchor_pos
                    spike_found = True
                    break

        if not spike_found:
            idx_pos += 1

    return clean_xy


def find_missing_spans(is_missing: np.ndarray) -> List[Tuple[int, int]]:
    """Find contiguous spans [start_idx, end_idx) of missing observations."""
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
    savgol_window: int = 7,
    savgol_polyorder: int = 2,
) -> BallTrackingResult:
    """Reconstruct missing ball positions across short occlusions and flag blackouts.

    Rules:
    - Missing gaps < max_gap_frames (1.4 s at 25 fps) with physically feasible endpoint
      velocities are smoothly interpolated.
    - Long blackouts (>= max_gap_frames) or gaps with physically impossible implied speeds
      (> 38 m/s) are NOT hallucinated; they are flagged as uncertain for downstream abstention ('X').
    - Continuous valid ball position segments are smoothed with a Savitzky-Golay filter to eliminate
      quantization jitter before velocity estimation.
    - Inside PCHIP interpolated gaps, ball velocity is computed using exact analytic derivatives.

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
    """
    n_frames = len(raw_ball_xy)
    dt = 1.0 / fps

    # Sanitize raw coordinates with bidirectional reachability
    cleaned_xy = sanitize_raw_ball_positions(
        raw_ball_xy,
        pitch=pitch,
        max_jump_speed_mps=max_physical_speed_mps + 2.0,
        fps=fps,
    )

    raw_missing = np.isnan(cleaned_xy[:, 0]) | np.isnan(cleaned_xy[:, 1])
    reconstructed_xy = cleaned_xy.copy()
    is_ball_uncertain = raw_missing.copy()
    ball_speed = np.full(n_frames, np.nan, dtype=np.float32)

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

        if np.isnan(p_prev[0]) or np.isnan(p_next[0]):
            is_ball_uncertain[start_idx:end_idx] = True
            continue

        time_gap = (gap_len + 1) * dt
        dx = p_next[0] - p_prev[0]
        dy = p_next[1] - p_prev[1]
        dist = np.sqrt(dx * dx + dy * dy)
        avg_speed = dist / time_gap

        if gap_len < max_gap_frames and avg_speed <= max_physical_speed_mps:
            gap_frame_indices = np.arange(start_idx, end_idx)

            if gap_len <= 4:
                # Linear interpolation
                alphas = (np.arange(1, gap_len + 1) / (gap_len + 1)).reshape(-1, 1)
                interp_points = p_prev + alphas * (p_next - p_prev)
                reconstructed_xy[start_idx:end_idx] = interp_points
                is_ball_uncertain[start_idx:end_idx] = False

                # Analytic constant speed for linear segment
                vx_linear = dx / time_gap
                vy_linear = dy / time_gap
                ball_speed[start_idx:end_idx] = float(np.sqrt(vx_linear * vx_linear + vy_linear * vy_linear))
            else:
                # Shape-preserving PCHIP with analytic derivatives
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
        else:
            is_ball_uncertain[start_idx:end_idx] = True

    # Smooth continuous valid segments using Savitzky-Golay filter
    valid_spans = find_missing_spans(is_ball_uncertain)
    # The valid spans are the complements of missing spans
    valid_blocks = []
    prev_end = 0
    for s, e in valid_spans:
        if s > prev_end:
            valid_blocks.append((prev_end, s))
        prev_end = e
    if prev_end < n_frames:
        valid_blocks.append((prev_end, n_frames))

    for vs, ve in valid_blocks:
        block_len = ve - vs
        if block_len >= savgol_window:
            # Apply light Savitzky-Golay filter along time axis
            reconstructed_xy[vs:ve, 0] = savgol_filter(
                reconstructed_xy[vs:ve, 0], window_length=savgol_window, polyorder=savgol_polyorder
            )
            reconstructed_xy[vs:ve, 1] = savgol_filter(
                reconstructed_xy[vs:ve, 1], window_length=savgol_window, polyorder=savgol_polyorder
            )

        # Compute central-difference velocity for frames in this block that don't have analytic speed
        for i in range(vs, ve):
            if np.isnan(ball_speed[i]):
                has_prev = (i > vs)
                has_next = (i < ve - 1)
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
