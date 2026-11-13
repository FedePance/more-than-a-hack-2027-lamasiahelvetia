"""Hysteresis possession state machine for robust possession spell identification.

Replaces noisy frame-by-frame nearest-player assignment with a temporal hysteresis
state machine to maximize spellF1 (possession continuity) and macroF1 across the 4 valid
labels ('H', 'A', 'D', 'X').

Core principles:
- Hysteresis: Transitions (H -> A or A -> H) require sustained control proximity (<= 1.8 m)
  for a minimum confirmation duration (>= 20-25 frames / 0.8-1.0 s).
- Inertia: In loose or contested balls, possession remains with the team that previously controlled it.
- Abstention ('X'): Emitted during prolonged tracking blackouts or contested scrambles so
  the official scorer's fill_abstentions naturally bridges possessions without fragmentation.
- Dead Ball ('D'): Cleanly overrides live possession during stoppages and non-playing periods.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional
import numpy as np


# Label definitions matching evaluation/possession_metrics.py
LABEL_H = "H"
LABEL_A = "A"
LABEL_D = "D"
LABEL_X = "X"
VALID_LABELS = {LABEL_H, LABEL_A, LABEL_D, LABEL_X}

HOME_CODE = 0
AWAY_CODE = 1


@dataclass
class PossessionResult:
    """Result of the possession engine."""
    labels: np.ndarray             # shape (n_frames,), dtype object ('H', 'A', 'D', 'X')
    min_dist_home: np.ndarray      # shape (n_frames,), float32 in metres
    min_dist_away: np.ndarray      # shape (n_frames,), float32 in metres
    current_possessor: np.ndarray  # shape (n_frames,), int8 (0 = Home, 1 = Away, -1 = Undefined)


def compute_team_distances_to_ball(
    player_xy: np.ndarray,
    ball_xy: np.ndarray,
    team_of_track: np.ndarray,
    is_ball_uncertain: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-frame minimum Euclidean distance from each team to the ball in metres.

    Args:
        player_xy: Array of player positions in metres, shape (n_frames, n_slots, 2).
        ball_xy: Array of ball positions in metres, shape (n_frames, 2).
        team_of_track: Array mapping slot index to team (0 = Home, 1 = Away), shape (n_slots,).
        is_ball_uncertain: Boolean array indicating whether ball tracking is missing/uncertain.

    Returns:
        Tuple of (min_dist_home, min_dist_away), both shape (n_frames,) float32.
    """
    n_frames = len(ball_xy)
    home_slots = np.flatnonzero(team_of_track == HOME_CODE)
    away_slots = np.flatnonzero(team_of_track == AWAY_CODE)

    min_dist_home = np.full(n_frames, 999.0, dtype=np.float32)
    min_dist_away = np.full(n_frames, 999.0, dtype=np.float32)

    known_ball = ~is_ball_uncertain & ~np.isnan(ball_xy[:, 0])
    valid_frames = np.flatnonzero(known_ball)

    if not len(valid_frames):
        return min_dist_home, min_dist_away

    # Compute distances for valid frames
    bx = ball_xy[valid_frames, 0:1]  # shape (V, 1)
    by = ball_xy[valid_frames, 1:2]  # shape (V, 1)

    # Home players
    if len(home_slots):
        hx = player_xy[valid_frames][:, home_slots, 0]  # shape (V, n_home)
        hy = player_xy[valid_frames][:, home_slots, 1]  # shape (V, n_home)
        dx_h = hx - bx
        dy_h = hy - by
        dist_sq_h = dx_h * dx_h + dy_h * dy_h
        # Nan-safe minimum
        with np.errstate(invalid="ignore"):
            min_sq_h = np.nanmin(dist_sq_h, axis=1)
        min_sq_h = np.nan_to_num(min_sq_h, nan=999.0 * 999.0)
        min_dist_home[valid_frames] = np.sqrt(np.maximum(0.0, min_sq_h)).astype(np.float32)

    # Away players
    if len(away_slots):
        ax = player_xy[valid_frames][:, away_slots, 0]  # shape (V, n_away)
        ay = player_xy[valid_frames][:, away_slots, 1]  # shape (V, n_away)
        dx_a = ax - bx
        dy_a = ay - by
        dist_sq_a = dx_a * dx_a + dy_a * dy_a
        with np.errstate(invalid="ignore"):
            min_sq_a = np.nanmin(dist_sq_a, axis=1)
        min_sq_a = np.nan_to_num(min_sq_a, nan=999.0 * 999.0)
        min_dist_away[valid_frames] = np.sqrt(np.maximum(0.0, min_sq_a)).astype(np.float32)

    return min_dist_home, min_dist_away


def identify_half_kickoff_team(
    half: int,
    kickoff_frame: int,
    smart_tagging_events: List[Dict[str, Any]],
    min_dist_home: np.ndarray,
    min_dist_away: np.ndarray,
) -> int:
    """Identify which team takes the kickoff for a given half.

    Home is Barcelona (0), Away is Athletic Club (1).
    """
    half_str = f"{half}st Half" if half == 1 else f"{half}nd Half"

    # Check Smart Tagging for kickoff event
    for event in smart_tagging_events:
        code = (event.get("code") or "").strip().lower()
        if code == "set pieces":
            event_type = (event.get("tags", {}).get("Type") or "").strip().lower()
            event_half = (event.get("tags", {}).get("Half") or "").strip().lower()
            if "kick off" in event_type and (event_half == half_str.lower() or event_half == f"{half}"):
                team_tag = (event.get("tags", {}).get("Team") or "").strip().lower()
                if "barcelona" in team_tag:
                    return HOME_CODE
                if "atletic" in team_tag or "athletic" in team_tag:
                    return AWAY_CODE

    # Physical fallback: closest team to the ball at the kickoff frame
    check_window = 10
    start = max(0, kickoff_frame)
    end = min(len(min_dist_home), kickoff_frame + check_window)
    h_dist = np.min(min_dist_home[start:end]) if end > start else 999.0
    a_dist = np.min(min_dist_away[start:end]) if end > start else 999.0

    return HOME_CODE if h_dist <= a_dist else AWAY_CODE


def run_possession_state_machine(
    n_frames: int,
    ball_xy: np.ndarray,
    is_ball_uncertain: np.ndarray,
    is_dead_ball: np.ndarray,
    player_xy: np.ndarray,
    player_speed: np.ndarray,
    team_of_track: np.ndarray,
    smart_tagging_events: List[Dict[str, Any]],
    half_blocks: List[Tuple[int, int, int]],
    control_radius: float = 1.8,
    min_control_frames: int = 22,
    scramble_abstain_frames: int = 15,
    fps: float = 25.0,
) -> PossessionResult:
    """Execute temporal hysteresis possession state machine across all frames.

    Args:
        n_frames: Total video frames (199,319).
        ball_xy: Reconstructed ball coordinates in metres, shape (n_frames, 2).
        is_ball_uncertain: Uncertainty mask for ball position, shape (n_frames,).
        is_dead_ball: Dead ball mask, shape (n_frames,).
        player_xy: Player positions in metres, shape (n_frames, n_slots, 2).
        player_speed: Player speeds in m/s, shape (n_frames, n_slots).
        team_of_track: Slot to team mapping (0 = Home, 1 = Away).
        smart_tagging_events: Parsed Smart Tagging events.
        half_blocks: List of (half, first_frame, last_frame) tuples from games.py.
        control_radius: Proximity threshold in metres to establish ball control (default: 1.8 m).
        min_control_frames: Consecutive frames required for opponent to take possession (default: 22 frames).
        scramble_abstain_frames: Consecutive contested frames before emitting 'X' (default: 15 frames).
        fps: Video frame rate.

    Returns:
        PossessionResult containing final labels array, team distances, and possessor history.
    """
    labels = np.full(n_frames, LABEL_D, dtype=object)
    possessor_history = np.full(n_frames, -1, dtype=np.int8)

    # Compute per-frame minimum distances from Home and Away to the ball
    min_dist_home, min_dist_away = compute_team_distances_to_ball(
        player_xy=player_xy,
        ball_xy=ball_xy,
        team_of_track=team_of_track,
        is_ball_uncertain=is_ball_uncertain,
    )

    # Process each half separately so state never bleeds across halftime
    for half, first_frame, last_frame in half_blocks:
        first_frame = max(0, first_frame)
        last_frame = min(n_frames - 1, last_frame)

        # Identify initial possessor for this half
        current_possessor = identify_half_kickoff_team(
            half=half,
            kickoff_frame=first_frame,
            smart_tagging_events=smart_tagging_events,
            min_dist_home=min_dist_home,
            min_dist_away=min_dist_away,
        )

        opp_control_streak = 0
        scramble_streak = 0
        was_dead_previous = True

        for t in range(first_frame, last_frame + 1):
            # 1. Dead Ball override
            if is_dead_ball[t]:
                labels[t] = LABEL_D
                possessor_history[t] = current_possessor
                opp_control_streak = 0
                scramble_streak = 0
                was_dead_previous = True
                continue

            # 2. Restart from dead ball: determine who restarted play
            if was_dead_previous:
                was_dead_previous = False
                # Inspect who touched or was nearest to the ball at the restart
                restart_window = 10
                t_end = min(last_frame + 1, t + restart_window)
                if t_end > t:
                    h_min = np.min(min_dist_home[t:t_end])
                    a_min = np.min(min_dist_away[t:t_end])
                    if h_min <= 2.5 or a_min <= 2.5:
                        current_possessor = HOME_CODE if h_min <= a_min else AWAY_CODE

            # 3. Ball tracking uncertain / blackout: emit 'X'
            if is_ball_uncertain[t]:
                labels[t] = LABEL_X
                possessor_history[t] = current_possessor
                opp_control_streak = 0
                scramble_streak = 0
                continue

            # 4. Live play with valid ball position
            d_home = min_dist_home[t]
            d_away = min_dist_away[t]

            # Distances for current possessing team and opposing team
            if current_possessor == HOME_CODE:
                d_possessor = d_home
                d_opponent = d_away
                opponent_team = AWAY_CODE
            else:
                d_possessor = d_away
                d_opponent = d_home
                opponent_team = HOME_CODE

            # Check for actively contested 50/50 duel / scramble
            is_contested = (
                (d_home <= control_radius)
                and (d_away <= control_radius)
                and (abs(d_home - d_away) < 0.4)
            )

            if is_contested:
                scramble_streak += 1
                if scramble_streak >= scramble_abstain_frames:
                    # Unresolved scramble: emit 'X' so spellF1 bridges the ongoing spell
                    labels[t] = LABEL_X
                    possessor_history[t] = current_possessor
                    opp_control_streak = 0
                    continue
            else:
                scramble_streak = 0

            # Opponent controlled possession check
            # Opponent must be within control radius AND closer than the current possessor
            opponent_has_control = (
                (d_opponent <= control_radius)
                and (d_opponent < d_possessor)
            )

            if opponent_has_control:
                opp_control_streak += 1
                if opp_control_streak >= min_control_frames:
                    # Turnover confirmed: hysteresis threshold met
                    current_possessor = opponent_team
                    opp_control_streak = 0
                    scramble_streak = 0
            else:
                # If opponent is well outside control radius, reset streak
                if d_opponent > control_radius + 0.6:
                    opp_control_streak = 0
                else:
                    # Leaky decay to tolerate minor tracking dropouts during continuous dribbling
                    opp_control_streak = max(0, opp_control_streak - 1)

            # Assign live possession label
            labels[t] = LABEL_H if current_possessor == HOME_CODE else LABEL_A
            possessor_history[t] = current_possessor

    # Final guarantee: all labels must strictly be in {'H', 'A', 'D', 'X'}
    invalid_mask = ~np.isin(labels, list(VALID_LABELS))
    if np.any(invalid_mask):
        labels[invalid_mask] = LABEL_D

    return PossessionResult(
        labels=labels,
        min_dist_home=min_dist_home,
        min_dist_away=min_dist_away,
        current_possessor=possessor_history,
    )
