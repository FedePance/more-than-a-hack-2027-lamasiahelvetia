"""Dead Ball ('D') detector combining tactical Smart Tagging annotations and player kinematics.

Detects when the ball is out of play or play is halted by combining:
1. Non-playing periods (pre-match, halftime, post-match) from official half boundaries.
2. Full Smart Tagging stoppage and set-piece intervals without premature cutoffs.
3. Out-of-bounds latching forward across subsequent uncertain frames until restart or stoppage end.
4. Standalone collective stoppage detection when 80th-percentile outfield player speed drops
   below walking/standing threshold (< 1.35 m/s) for at least 60 frames (2.4 s).
5. Kinematic stoppage within dilated stoppage windows or around stationary abandoned balls.
"""

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.ndimage import binary_dilation

from src.mask_utils import fill_short_gaps, remove_short_spans

# Out-of-bounds tolerance (m) and proximity (m) that counts as a controlled restart
_BOUNDARY_MARGIN_M = 0.6
_RESTART_CONTROL_DISTANCE_M = 2.2
# Sentinel distance (m) used when no player position is available
_NO_PLAYER_DISTANCE_M = 999.0
# Temporal regularization windows (frames at 25 fps)
_SPEED_SMOOTHING_FRAMES = 25
_BRIDGE_GAP_FRAMES = 15


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
    "stoppages",
    "substitution",
    "substitutions",
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
    """Apply uniform moving average smoothing to a 1D signal (same length, float32 output)."""
    if len(signal) == 0 or window_size <= 1:
        return np.array(signal, dtype=np.float32)

    # Asymmetric padding keeps the output length equal to the input for even windows too
    pad_left = window_size // 2
    pad_right = window_size - 1 - pad_left
    padded = np.pad(signal, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window_size, dtype=np.float32) / window_size
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed.astype(np.float32)


def is_stoppage_event(event: Dict[str, Any]) -> bool:
    """Whether a Smart Tagging event denotes a stoppage or a set-piece setup interval."""
    code = (event.get("code") or "").strip().lower()
    if code in STOPPAGE_CODES:
        return True
    if "set piece" in code:
        event_type = ((event.get("tags") or {}).get("Type") or "").strip().lower()
        return not event_type or any(t in event_type for t in STOPPAGE_SET_PIECE_TYPES)
    return False


def build_stoppage_tag_mask(smart_tagging_events: List[Dict[str, Any]], n_frames: int) -> np.ndarray:
    """Rasterize Smart Tagging stoppage / set-piece intervals onto the video frame axis.

    Args:
        smart_tagging_events: Parsed events with half-open [start_frame, end_frame) intervals.
        n_frames: Total number of video frames.

    Returns:
        Boolean mask of shape (n_frames,), True inside any stoppage interval.
    """
    mask = np.zeros(n_frames, dtype=bool)
    for event in smart_tagging_events:
        start_frame = max(0, int(event.get("start_frame") or 0))
        end_frame = min(n_frames, int(event.get("end_frame") or 0))
        if start_frame < end_frame and is_stoppage_event(event):
            mask[start_frame:end_frame] = True
    return mask


def _clamp_half_blocks(half_blocks: List[Tuple[int, int, int]], n_frames: int) -> List[Tuple[int, int]]:
    """Convert inclusive (half, first, last) blocks to clamped half-open [start, end) spans."""
    spans = []
    for _, first_frame, last_frame in half_blocks:
        start = max(0, first_frame)
        end = min(n_frames, last_frame + 1)
        if start < end:
            spans.append((start, end))
    return spans


def _latch_out_of_bounds(
    is_dead: np.ndarray,
    out_of_bounds: np.ndarray,
    tag_stoppage_mask: np.ndarray,
    controlled_in_bounds: np.ndarray,
    start: int,
    end: int,
) -> None:
    """Mark dead-ball frames from each out-of-bounds excursion until play restarts (in place).

    Once the ball is seen out of bounds, the dead state is held across the following frames
    (including blackout frames) until either an overlapping Smart Tagging stoppage ends or the
    ball is back in bounds under player control. Only frames inside [start, end) are touched,
    so the latch never crosses a half boundary.
    """
    oob_frames = start + np.flatnonzero(out_of_bounds[start:end])
    cursor = 0
    while cursor < len(oob_frames):
        t = int(oob_frames[cursor])
        seen_stoppage = False

        while t < end:
            if out_of_bounds[t]:
                # A fresh excursion restarts the wait for an overlapping stoppage
                seen_stoppage = False
            if tag_stoppage_mask[t]:
                seen_stoppage = True
            elif seen_stoppage:
                break  # the overlapping Smart Tagging stoppage has ended
            if controlled_in_bounds[t]:
                break  # ball back in play under player control
            is_dead[t] = True
            t += 1

        # While unlatched nothing happens until the next out-of-bounds frame
        cursor = int(np.searchsorted(oob_frames, t))


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
    standalone_stoppage_frames: int = 60,
    standalone_p80_threshold_mps: float = 1.35,
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
        standalone_stoppage_frames: Consecutive frames of low collective speed for standalone D (default: 60 frames / 2.4 s).
        standalone_p80_threshold_mps: Outfield 80th-percentile speed threshold for standalone D (default: 1.35 m/s).
        fps: Video frame rate (unused; all windows are expressed in frames).

    Returns:
        DeadBallResult container with is_dead_ball boolean mask and speed series.

    Raises:
        ValueError: If a per-frame input does not have n_frames rows or a window is not positive.
    """
    per_frame_inputs = {
        "mean_player_speed": mean_player_speed,
        "p80_player_speed": p80_player_speed,
        "ball_xy": ball_xy,
        "is_ball_uncertain": is_ball_uncertain,
        "ball_speed": ball_speed,
    }
    for name, values in per_frame_inputs.items():
        if len(values) != n_frames:
            raise ValueError(f"{name} has {len(values)} rows, expected n_frames={n_frames}")
    if dilation_window_frames < 1:
        raise ValueError(f"dilation_window_frames must be >= 1, got {dilation_window_frames}")

    is_ball_uncertain = np.asarray(is_ball_uncertain, dtype=bool)
    half_spans = _clamp_half_blocks(half_blocks, n_frames)

    # 1. Non-playing periods outside official halves are definitely dead ball
    in_half = np.zeros(n_frames, dtype=bool)
    for s, e in half_spans:
        in_half[s:e] = True
    is_dead = ~in_half

    # 2. Smart Tagging intervals for stoppages and set pieces
    # Preserve full [start_frame, end_frame) interval for all stoppage tags
    tag_stoppage_mask = build_stoppage_tag_mask(smart_tagging_events, n_frames)
    is_dead |= tag_stoppage_mask

    # 3. Morphological dilation around Smart Tagging stoppages
    dilation_structure = np.ones(dilation_window_frames, dtype=bool)
    dilated_stoppage_window = binary_dilation(tag_stoppage_mask, structure=dilation_structure)

    # 4. Pitch boundary excursions & out-of-bounds latching
    length, width = pitch
    known_ball = ~is_ball_uncertain & ~np.isnan(ball_xy[:, 0])

    out_of_bounds = known_ball & (
        (ball_xy[:, 0] < -_BOUNDARY_MARGIN_M)
        | (ball_xy[:, 0] > length + _BOUNDARY_MARGIN_M)
        | (ball_xy[:, 1] < -_BOUNDARY_MARGIN_M)
        | (ball_xy[:, 1] > width + _BOUNDARY_MARGIN_M)
    )

    # Compute distance from ball to closest player
    min_player_dist = np.full(n_frames, _NO_PLAYER_DISTANCE_M, dtype=np.float32)
    if player_xy is not None and len(player_xy) == n_frames and player_xy.shape[1] > 0:
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN frames
            dists_sq = (player_xy[:, :, 0] - ball_xy[:, 0:1]) ** 2 + (player_xy[:, :, 1] - ball_xy[:, 1:2]) ** 2
            min_sq = np.nanmin(dists_sq, axis=1)
        min_player_dist = np.sqrt(
            np.nan_to_num(min_sq, nan=_NO_PLAYER_DISTANCE_M * _NO_PLAYER_DISTANCE_M)
        ).astype(np.float32)

    in_bounds = known_ball & ~out_of_bounds
    controlled_in_bounds = in_bounds & (min_player_dist <= _RESTART_CONTROL_DISTANCE_M)

    # Latch out-of-bounds forward across subsequent uncertain frames until controlled restart
    # or stoppage end, independently inside each half
    for s, e in half_spans:
        _latch_out_of_bounds(is_dead, out_of_bounds, tag_stoppage_mask, controlled_in_bounds, s, e)

    # 5. Outfield player kinematics: 80th-percentile speed
    smoothed_p80 = smooth_signal(p80_player_speed, window_size=_SPEED_SMOOTHING_FRAMES)
    smoothed_mean = smooth_signal(mean_player_speed, window_size=_SPEED_SMOOTHING_FRAMES)

    # Condition (a): 80% of outfield players moving slower than jogging threshold (< 1.4 m/s)
    is_low_p80_speed = smoothed_p80 < jogging_speed_threshold_mps

    # Condition (b): ball is visible, stationary (< 0.35 m/s), and abandoned (no player within 2.5 m)
    with np.errstate(invalid="ignore"):
        stationary_ball = known_ball & (ball_speed < stationary_ball_speed_mps)
    abandoned_ball = stationary_ball & (min_player_dist > abandoned_ball_distance_m)

    # Trigger kinematic D when p80 < 1.4 m/s sustained for >= 50 frames AND (dilated stoppage OR abandoned ball)
    kinematic_candidate = is_low_p80_speed & (dilated_stoppage_window | abandoned_ball)
    is_dead |= remove_short_spans(kinematic_candidate, min_length=kinematic_stoppage_frames)

    # Standalone collective stoppage: even outside Smart Tagging dilation windows,
    # if p80 outfield player speed drops below standalone_p80_threshold_mps (1.35 m/s)
    # for >= standalone_stoppage_frames (60 frames / 2.4 s), mark as Dead Ball (D)
    is_standalone_p80 = (p80_player_speed < standalone_p80_threshold_mps) | (smoothed_p80 < standalone_p80_threshold_mps)
    is_dead |= remove_short_spans(is_standalone_p80, min_length=standalone_stoppage_frames)

    # 6. Temporal regularization
    # Bridge small gaps (<= 15 frames / 0.6 s) between contiguous stoppages
    is_dead = fill_short_gaps(is_dead, max_gap=_BRIDGE_GAP_FRAMES)

    # Remove sub-second micro-blips during live halves
    for s, e in half_spans:
        is_dead[s:e] = remove_short_spans(is_dead[s:e], min_length=min_stoppage_duration_frames)

    # Outside official halves must always remain dead ball
    is_dead[~in_half] = True

    return DeadBallResult(
        is_dead_ball=is_dead,
        smoothed_player_speed=smoothed_mean,
        p80_player_speed=smoothed_p80,
    )
