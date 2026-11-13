"""Dead Ball ('D') detector combining tactical Smart Tagging annotations and player kinematics.

Detects when the ball is out of play or play is halted by combining:
1. Non-playing periods (pre-match, halftime, post-match) from official half boundaries.
2. Smart Tagging stoppage and set-piece setup intervals, releasing to live play when the restart kick is struck.
3. True morphological dilation (scipy.ndimage.binary_dilation) for stoppage contexts.
4. Physical kinematics: outfield players' 80th-percentile speed dropping below jogging threshold (< 1.4 m/s)
   sustained for >= 50 frames (2.0 s) within dilated stoppage windows or around stationary abandoned balls.
5. Pitch boundary excursions where the ball physically leaves the pitch.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
from scipy.ndimage import binary_dilation


@dataclass
class DeadBallResult:
    """Result of dead ball detection."""
    is_dead_ball: np.ndarray            # shape (n_frames,), bool (True where play is dead / halted)
    smoothed_player_speed: np.ndarray   # shape (n_frames,), float32 in m/s (smoothed collective speed)
    p80_player_speed: np.ndarray        # shape (n_frames,), float32 in m/s (smoothed 80th percentile)


# Stoppage codes in Smart Tagging (case-insensitive)
STOPPAGE_CODES = {
    "set pieces",
    "goals",
    "foul",
    "fouls",
    "offside",
    "offsides",
    "penalty",
    "penalties",
    "stoppage",
    "substitution",
}

STOPPAGE_SET_PIECE_TYPES = {
    "throw-in",
    "corner kick",
    "goal kick",
    "free kick",
    "kick off (start)",
    "kick off (after goal)",
    "penalty kick",
    "penalty",
}


def smooth_signal(signal: np.ndarray, window_size: int = 35) -> np.ndarray:
    """Apply uniform moving average smoothing to a 1D signal."""
    if len(signal) == 0 or window_size <= 1:
        return signal.copy()

    pad_width = window_size // 2
    padded = np.pad(signal, pad_width, mode="edge")
    kernel = np.ones(window_size, dtype=np.float32) / window_size
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed.astype(np.float32)


def remove_short_spans(mask: np.ndarray, min_length: int) -> np.ndarray:
    """Remove True spans shorter than min_length from a boolean mask (binary opening)."""
    out = mask.copy()
    diffs = np.diff(out.astype(np.int8))
    starts = (np.flatnonzero(diffs == 1) + 1).tolist()
    ends = (np.flatnonzero(diffs == -1) + 1).tolist()

    if out[0]:
        starts = [0] + starts
    if out[-1]:
        ends = ends + [len(out)]

    for start, end in zip(starts, ends):
        if end - start < min_length:
            out[start:end] = False
    return out


def fill_short_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill False gaps between True regions shorter than max_gap (binary closing)."""
    out = mask.copy()
    diffs = np.diff((~out).astype(np.int8))
    starts = (np.flatnonzero(diffs == 1) + 1).tolist()
    ends = (np.flatnonzero(diffs == -1) + 1).tolist()

    if not out[0]:
        starts = [0] + starts
    if not out[-1]:
        ends = ends + [len(out)]

    for start, end in zip(starts, ends):
        if start > 0 and end < len(out):
            if end - start <= max_gap:
                out[start:end] = True
    return out


def detect_dead_ball(
    n_frames: int,
    mean_player_speed: np.ndarray,
    p80_player_speed: np.ndarray,
    ball_xy: np.ndarray,
    is_ball_uncertain: np.ndarray,
    ball_speed: np.ndarray,
    player_xy: np.ndarray,
    smart_tagging_events: List[Dict[str, Any]],
    half_blocks: List[Tuple[int, int, int]],
    pitch: Tuple[float, float] = (105.3, 68.0),
    jogging_speed_threshold_mps: float = 1.40,
    kinematic_stoppage_frames: int = 50,
    dilation_window_frames: int = 125,
    min_stoppage_duration_frames: int = 20,
    stationary_ball_speed_mps: float = 0.35,
    abandoned_ball_distance_m: float = 2.5,
    fps: float = 25.0,
) -> DeadBallResult:
    """Detect dead-ball frames ('D') combining Smart Tagging and outfield kinematics.

    Args:
        n_frames: Total number of frames in the match recording (e.g. 199,319).
        mean_player_speed: Collective average outfield player speed per frame, shape (n_frames,).
        p80_player_speed: 80th-percentile outfield player speed per frame, shape (n_frames,).
        ball_xy: Reconstructed ball coordinates in metres, shape (n_frames, 2).
        is_ball_uncertain: Uncertainty mask for ball position, shape (n_frames,).
        ball_speed: Ball speed per frame in m/s, shape (n_frames,).
        player_xy: Player coordinates in metres, shape (n_frames, n_slots, 2).
        smart_tagging_events: Parsed events from Smart Tagging XML.
        half_blocks: List of (half, first_frame, last_frame) tuples from games.py.
        pitch: Pitch dimensions in metres (length, width).
        jogging_speed_threshold_mps: Outfield 80th-percentile speed threshold for kinematic D (default: 1.40 m/s).
        kinematic_stoppage_frames: Minimum consecutive frames of low speed required (default: 50 frames / 2.0 s).
        dilation_window_frames: Window size for morphological dilation around stoppages (default: 125 frames / 5.0 s).
        min_stoppage_duration_frames: Minimum valid stoppage duration in frames (default: 20 frames / 0.8 s).
        stationary_ball_speed_mps: Ball considered stationary below this speed (default: 0.35 m/s).
        abandoned_ball_distance_m: Minimum player distance to flag abandoned stationary ball (default: 2.5 m).
        fps: Video frame rate.

    Returns:
        DeadBallResult container with is_dead_ball boolean mask and speed series.
    """
    is_dead = np.zeros(n_frames, dtype=bool)
    tag_stoppage_mask = np.zeros(n_frames, dtype=bool)

    # 1. Non-playing periods outside official halves are definitely dead ball
    in_half = np.zeros(n_frames, dtype=bool)
    for _, start_frame, end_frame in half_blocks:
        s = max(0, start_frame)
        e = min(n_frames, end_frame + 1)
        in_half[s:e] = True
    is_dead[~in_half] = True

    # 2. Smart Tagging intervals for stoppages and set pieces
    for event in smart_tagging_events:
        code = (event.get("code") or "").strip().lower()
        start_frame = max(0, event.get("start_frame", 0))
        end_frame = min(n_frames, event.get("end_frame", 0))

        if start_frame >= end_frame:
            continue

        is_stoppage = False
        if code in STOPPAGE_CODES:
            is_stoppage = True
        elif "set piece" in code:
            event_type = (event.get("tags", {}).get("Type") or "").strip().lower()
            if not event_type or any(t in event_type for t in STOPPAGE_SET_PIECE_TYPES):
                is_stoppage = True

        if is_stoppage:
            tag_stoppage_mask[start_frame:end_frame] = True

            # Refine end of set piece: if ball is struck into live flight (speed > 3.0 m/s)
            # towards the end of the tagged window, release dead ball at the kick frame
            stoppage_end = end_frame
            check_start = max(start_frame + int(fps * 1.5), end_frame - int(fps * 1.5))
            if check_start < end_frame:
                for f in range(check_start, end_frame):
                    if not is_ball_uncertain[f] and ball_speed[f] > 3.0:
                        stoppage_end = f
                        break
            is_dead[start_frame:stoppage_end] = True

    # 3. Morphological dilation around Smart Tagging stoppages
    dilation_structure = np.ones(dilation_window_frames, dtype=bool)
    dilated_stoppage_window = binary_dilation(tag_stoppage_mask, structure=dilation_structure)

    # 4. Pitch boundary excursions
    length, width = pitch
    boundary_margin = 0.6
    known_ball = ~is_ball_uncertain & ~np.isnan(ball_xy[:, 0])

    out_of_bounds = known_ball & (
        (ball_xy[:, 0] < -boundary_margin)
        | (ball_xy[:, 0] > length + boundary_margin)
        | (ball_xy[:, 1] < -boundary_margin)
        | (ball_xy[:, 1] > width + boundary_margin)
    )
    is_dead[out_of_bounds] = True

    # 5. Outfield player kinematics: 80th-percentile speed
    # Smooth p80 speed over 25 frames
    smoothed_p80 = smooth_signal(p80_player_speed, window_size=25)
    smoothed_mean = smooth_signal(mean_player_speed, window_size=25)

    # Condition: 80% of outfield players are moving slower than jogging threshold (< 1.4 m/s)
    is_low_p80_speed = smoothed_p80 < jogging_speed_threshold_mps

    # Condition (b): ball is visible, stationary (< 0.35 m/s), and abandoned (no player within 2.5 m)
    stationary_ball = known_ball & (ball_speed < stationary_ball_speed_mps)

    # Compute distance from ball to closest player
    abandoned_ball = np.zeros(n_frames, dtype=bool)
    if player_xy is not None and len(player_xy) == n_frames:
        for f in np.flatnonzero(stationary_ball):
            bx = ball_xy[f, 0]
            by = ball_xy[f, 1]
            px = player_xy[f, :, 0]
            py = player_xy[f, :, 1]
            dists_sq = (px - bx) ** 2 + (py - by) ** 2
            with np.errstate(invalid="ignore"):
                min_dist = np.sqrt(np.nanmin(dists_sq)) if np.any(~np.isnan(dists_sq)) else 999.0
            if min_dist > abandoned_ball_distance_m:
                abandoned_ball[f] = True

    # Trigger kinematic D only when p80 < 1.4 m/s sustained for >= 50 frames AND (dilated stoppage OR abandoned ball)
    kinematic_candidate = is_low_p80_speed & (dilated_stoppage_window | abandoned_ball)
    sustained_kinematic_stoppage = remove_short_spans(kinematic_candidate, min_length=kinematic_stoppage_frames)
    is_dead = is_dead | sustained_kinematic_stoppage

    # 6. Temporal regularization
    # Bridge small gaps (< 15 frames / 0.6 s) between contiguous stoppages
    is_dead = fill_short_gaps(is_dead, max_gap=15)

    # Remove sub-second micro-blips during live halves
    for _, start_frame, end_frame in half_blocks:
        s = max(0, start_frame)
        e = min(n_frames, end_frame + 1)
        half_mask = is_dead[s:e]
        cleaned_half_mask = remove_short_spans(half_mask, min_length=min_stoppage_duration_frames)
        is_dead[s:e] = cleaned_half_mask

    # Outside official halves must always remain dead ball
    is_dead[~in_half] = True

    return DeadBallResult(
        is_dead_ball=is_dead,
        smoothed_player_speed=smoothed_mean,
        p80_player_speed=smoothed_p80,
    )
