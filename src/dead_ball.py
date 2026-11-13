"""Dead Ball ('D') detector combining tactical Smart Tagging annotations and player kinematics.

Detects when the ball is out of play or play is halted by combining:
1. Non-playing periods (pre-match, halftime, post-match) defined in games.py
2. Smart Tagging stoppage and set-piece setup intervals (fouls, throw-ins, corners, goal kicks, free kicks, goals)
3. Pitch boundary excursions where the ball leaves the physical pitch limits
4. Collective outfield player kinematics: smoothed collective walking/standing speed (< 1.15 m/s)
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional
import numpy as np


@dataclass
class DeadBallResult:
    """Result of dead ball detection."""
    is_dead_ball: np.ndarray            # shape (n_frames,), bool (True where play is dead / halted)
    smoothed_player_speed: np.ndarray   # shape (n_frames,), float32 in m/s (smoothed collective speed)


# Stoppage codes in Smart Tagging (case-insensitive)
STOPPAGE_CODES = {
    "set pieces",
    "goals",
}

# Set piece types that represent dead ball setup until restart
STOPPAGE_SET_PIECE_TYPES = {
    "throw-in",
    "corner kick",
    "goal kick",
    "free kick",
    "kick off (start)",
    "kick off (after goal)",
}


def smooth_signal(signal: np.ndarray, window_size: int = 35) -> np.ndarray:
    """Apply uniform moving average smoothing to a 1D signal.

    Args:
        signal: 1D array of values.
        window_size: Window length in frames (must be odd).

    Returns:
        Smoothed array of identical shape.
    """
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
        # Only fill gaps between two True sections (not leading/trailing boundaries)
        if start > 0 and end < len(out):
            if end - start <= max_gap:
                out[start:end] = True
    return out


def detect_dead_ball(
    n_frames: int,
    mean_player_speed: np.ndarray,
    ball_xy: np.ndarray,
    is_ball_uncertain: np.ndarray,
    ball_speed: np.ndarray,
    smart_tagging_events: List[Dict[str, Any]],
    half_blocks: List[Tuple[int, int, int]],
    pitch: Tuple[float, float] = (105.3, 68.0),
    walking_speed_threshold_mps: float = 1.15,
    kinematics_window: int = 35,
    min_stoppage_duration_frames: int = 20,
    fps: float = 25.0,
) -> DeadBallResult:
    """Detect dead-ball frames ('D') across all video frames.

    Args:
        n_frames: Total number of frames in the match recording (e.g. 199,319).
        mean_player_speed: Collective average outfield player speed per frame, shape (n_frames,).
        ball_xy: Reconstructed ball coordinates in metres, shape (n_frames, 2).
        is_ball_uncertain: Uncertainty mask for ball position, shape (n_frames,).
        ball_speed: Ball speed per frame in m/s, shape (n_frames,).
        smart_tagging_events: Parsed events from Smart Tagging XML.
        half_blocks: List of (half, first_frame, last_frame) tuples from games.py.
        pitch: Pitch dimensions in metres (length, width).
        walking_speed_threshold_mps: Collective speed threshold below which players are walking/static.
        kinematics_window: Smoothing window in frames for player speed.
        min_stoppage_duration_frames: Minimum duration for a physically valid stoppage.
        fps: Video frame rate.

    Returns:
        DeadBallResult container with is_dead_ball boolean mask and smoothed collective speed.
    """
    is_dead = np.zeros(n_frames, dtype=bool)

    # 1. Non-playing periods (outside the two official halves) are definitely dead ball
    in_half = np.zeros(n_frames, dtype=bool)
    for _, start_frame, end_frame in half_blocks:
        s = max(0, start_frame)
        e = min(n_frames, end_frame + 1)
        in_half[s:e] = True
    is_dead[~in_half] = True

    # 2. Smart Tagging intervals for set pieces and goals
    for event in smart_tagging_events:
        code = (event.get("code") or "").strip().lower()
        start_frame = max(0, event.get("start_frame", 0))
        end_frame = min(n_frames, event.get("end_frame", 0))

        if start_frame >= end_frame:
            continue

        if code == "set pieces":
            event_type = (event.get("tags", {}).get("Type") or "").strip().lower()
            if not event_type or event_type in STOPPAGE_SET_PIECE_TYPES:
                # Set piece preparation interval is a dead ball stoppage
                is_dead[start_frame:end_frame] = True
        elif code == "goals":
            # Goal stoppage until play restarts
            is_dead[start_frame:end_frame] = True

    # 3. Ball out of physical pitch boundaries (with 0.6 m tolerance margin)
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

    # 4. Outfield player kinematics: collective walking/standing speed
    smoothed_speed = smooth_signal(mean_player_speed, window_size=kinematics_window)

    # Low collective speed indicates halted play (walking/standing)
    # When players collectively move at < walking threshold for sustained intervals in play
    is_low_collective_speed = smoothed_speed < walking_speed_threshold_mps

    # To avoid flagging momentary slow-downs during live passing, combine low speed with:
    # (a) ball missing/uncertain, or
    # (b) ball stationary (speed < 0.6 m/s), or
    # (c) within 5 seconds of a set piece / out of bounds
    stoppage_context = is_dead.copy()
    # Dilate stoppage context by 125 frames (5 seconds) to catch deceleration into stoppages
    stoppage_context = fill_short_gaps(stoppage_context, max_gap=125)

    stationary_ball = known_ball & (ball_speed < 0.6)
    kinematic_stoppage = is_low_collective_speed & (is_ball_uncertain | stationary_ball | stoppage_context)

    # Require kinematic stoppage to be sustained for at least 25 frames (1.0 s)
    sustained_kinematic_stoppage = remove_short_spans(kinematic_stoppage, min_length=int(fps))
    is_dead = is_dead | sustained_kinematic_stoppage

    # 5. Temporal morphology: remove spurious micro-blips during live halves
    # First bridge small gaps (< 15 frames / 0.6 s) between contiguous stoppages
    is_dead = fill_short_gaps(is_dead, max_gap=15)

    # Remove sub-second dead ball blips during live gameplay
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
        smoothed_player_speed=smoothed_speed,
    )
