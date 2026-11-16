"""Hysteresis possession state machine for robust possession spell identification.

Combines:
1. Dynamic team name resolution matching EPTS tracking metadata against Smart Tagging tags.
2. Smart restart attribution prioritizing Smart Tagging 'Team' tags and 50-frame forward search.
3. Pass-preserving possession evidence accumulation (opp_evidence) with free-flight freezing.
4. Minimum live spell regularization absorbing brief micro-flips (< 35 frames) into the surrounding spell.
5. Selective abstention ('X') during unresolved contested scrambles or unestablished restart blackouts,
   while maintaining established possession during occluded dribbles.
"""

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.mask_utils import find_true_runs

LABEL_H = "H"
LABEL_A = "A"
LABEL_D = "D"
LABEL_X = "X"
VALID_LABELS = {LABEL_H, LABEL_A, LABEL_D, LABEL_X}

HOME_CODE = 0
AWAY_CODE = 1

# Sentinel distance (m) used when the ball or every player of a team is unobserved
NO_DISTANCE_M = 999.0

# Fixed state-machine geometry (metres) and timing (frames at 25 fps)
_POSSESSOR_RESET_RADIUS_M = 1.8      # possessor this close (and closest) clears opponent evidence
_OPPONENT_EVIDENCE_RADIUS_M = 2.0    # opponent this close (and closest) accumulates evidence
_CONTESTED_MARGIN_M = 0.4            # both teams in range and this evenly matched = duel
_RESTART_TAKER_RADIUS_M = 2.5        # restart taker must be this close to the ball
_RESTART_TAG_SLACK_FRAMES = 50       # Smart Tagging event may sit this far (2.0 s) from a restart
_ESTABLISH_CONTROL_FRAMES = 15       # sustained proximity that confirms possession
_KICKOFF_CHECK_FRAMES = 5            # frames inspected for the physical kickoff fallback


@dataclass
class PossessionResult:
    """Result of the possession engine."""
    labels: np.ndarray             # shape (n_frames,), dtype object ('H', 'A', 'D', 'X')
    min_dist_home: np.ndarray      # shape (n_frames,), float32 in metres
    min_dist_away: np.ndarray      # shape (n_frames,), float32 in metres
    current_possessor: np.ndarray  # shape (n_frames,), int8 (0 = Home, 1 = Away, -1 = Undefined)


def match_team_string(team_str: str, home_name: str, away_name: str) -> Optional[int]:
    """Dynamically match a team tag string against Home and Away club names.

    Args:
        team_str: String from Smart Tagging (e.g. 'Barcelona', 'Atletic Club').
        home_name: Home club name from metadata (e.g. 'FC Barcelona').
        away_name: Away club name from metadata (e.g. 'Athletic Club').

    Returns:
        HOME_CODE (0), AWAY_CODE (1), or None if unresolved.
    """
    if not team_str or team_str.strip().upper() == "N/A":
        return None

    clean_tag = team_str.lower().replace(" ", "").replace("-", "")
    clean_home = (home_name or "").lower().replace(" ", "").replace("-", "")
    clean_away = (away_name or "").lower().replace(" ", "").replace("-", "")
    if not clean_tag:
        return None

    # Exact or substring containment (an empty club name must never match everything)
    if clean_home and (clean_tag in clean_home or clean_home in clean_tag):
        return HOME_CODE
    if clean_away and (clean_tag in clean_away or clean_away in clean_tag):
        return AWAY_CODE

    # Common aliases / roots (e.g. 'barca', 'athletic', 'atletic')
    if "barca" in clean_tag or "barce" in clean_tag:
        return HOME_CODE
    if "athle" in clean_tag or "atlet" in clean_tag or "bilbao" in clean_tag:
        return AWAY_CODE

    return None


def compute_team_distances_to_ball(
    player_xy: np.ndarray,
    ball_xy: np.ndarray,
    team_of_track: np.ndarray,
    is_ball_uncertain: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-frame minimum Euclidean distance from each team to the ball in metres.

    Frames with an unknown ball, and teams with no observed player, get NO_DISTANCE_M (999.0).
    """
    n_frames = len(ball_xy)
    known_ball = ~np.asarray(is_ball_uncertain, dtype=bool) & ~np.isnan(ball_xy).any(axis=1)
    valid_frames = np.flatnonzero(known_ball)
    ball_valid = ball_xy[valid_frames]

    def team_min_distance(team_code: int) -> np.ndarray:
        min_dist = np.full(n_frames, NO_DISTANCE_M, dtype=np.float32)
        slots = np.flatnonzero(team_of_track == team_code)
        if not len(valid_frames) or not len(slots):
            return min_dist

        team_xy = player_xy[np.ix_(valid_frames, slots)]  # (n_valid, n_team_slots, 2)
        dx = team_xy[:, :, 0] - ball_valid[:, 0:1]
        dy = team_xy[:, :, 1] - ball_valid[:, 1:2]
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # frames with no tracked player
            min_sq = np.nanmin(dx * dx + dy * dy, axis=1)
        min_sq = np.nan_to_num(min_sq, nan=NO_DISTANCE_M * NO_DISTANCE_M)
        min_dist[valid_frames] = np.sqrt(np.maximum(0.0, min_sq)).astype(np.float32)
        return min_dist

    return team_min_distance(HOME_CODE), team_min_distance(AWAY_CODE)


def identify_half_kickoff_team(
    half: int,
    kickoff_frame: int,
    smart_tagging_events: List[Dict[str, Any]],
    team_names: Tuple[str, str],
    min_dist_home: np.ndarray,
    min_dist_away: np.ndarray,
) -> int:
    """Identify which team takes the kickoff for a given half using dynamic name matching."""
    home_name, away_name = team_names
    half_str = f"{half}st Half" if half == 1 else f"{half}nd Half"

    # Match Smart Tagging kickoff event
    for event in smart_tagging_events:
        code = (event.get("code") or "").strip().lower()
        if code == "set pieces":
            tags = event.get("tags") or {}
            event_type = (tags.get("Type") or "").strip().lower()
            event_half = (tags.get("Half") or "").strip().lower()
            if "kick off" in event_type and (event_half == half_str.lower() or event_half == str(half)):
                matched = match_team_string(tags.get("Team") or "", home_name, away_name)
                if matched is not None:
                    return matched

    # Physical fallback: player closest to ball at kickoff frame
    start = max(0, kickoff_frame)
    end = min(len(min_dist_home), kickoff_frame + _KICKOFF_CHECK_FRAMES)
    h_dist = np.min(min_dist_home[start:end]) if end > start else NO_DISTANCE_M
    a_dist = np.min(min_dist_away[start:end]) if end > start else NO_DISTANCE_M

    return HOME_CODE if h_dist <= a_dist else AWAY_CODE


def attribute_restart_possession(
    frame_idx: int,
    smart_tagging_events: List[Dict[str, Any]],
    team_names: Tuple[str, str],
    min_dist_home: np.ndarray,
    min_dist_away: np.ndarray,
    is_ball_uncertain: np.ndarray,
    search_forward_frames: int = 50,
) -> Optional[int]:
    """Attribute initial possession at a dead ball restart.

    1. First checks the Team tag of the first Smart Tagging event (in file order) whose
       interval, widened by 50 frames (2.0 s) on both sides, contains the restart frame.
    2. Physical fallback: searches forward up to search_forward_frames (2.0 s) for the first
       frame where a player is close to the ball (<= 2.5 m) to identify the restart taker.
    """
    home_name, away_name = team_names

    # Check Smart Tagging events near this restart
    for event in smart_tagging_events:
        start_f = event.get("start_frame") or 0
        end_f = event.get("end_frame") or 0
        if start_f - _RESTART_TAG_SLACK_FRAMES <= frame_idx <= end_f + _RESTART_TAG_SLACK_FRAMES:
            tag_team = (event.get("tags") or {}).get("Team") or ""
            matched = match_team_string(tag_team, home_name, away_name)
            if matched is not None:
                return matched

    # Physical fallback: first frame at restart where the ball is visible next to a player
    search_end = min(len(min_dist_home), frame_idx + search_forward_frames)
    if search_end <= frame_idx:
        return None

    window = slice(frame_idx, search_end)
    h_dist = min_dist_home[window]
    a_dist = min_dist_away[window]
    near_player = ~np.asarray(is_ball_uncertain[window], dtype=bool) & (
        (h_dist <= _RESTART_TAKER_RADIUS_M) | (a_dist <= _RESTART_TAKER_RADIUS_M)
    )
    hits = np.flatnonzero(near_player)
    if not len(hits):
        return None
    first = hits[0]
    return HOME_CODE if h_dist[first] <= a_dist[first] else AWAY_CODE


def regularize_live_spells(
    labels: np.ndarray,
    half_blocks: List[Tuple[int, int, int]],
    min_live_spell_frames: int = 35,
    possessor_history: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Absorb brief H->A->H or A->H->A micro-flips within continuous live-play segments.

    Within each continuous live-play segment (bounded by 'D' frames and half boundaries),
    absorbs any brief micro-flip lasting fewer than min_live_spell_frames into the
    surrounding team's possession spell to prevent transient 1-touch clearances from
    fragmenting ongoing spells.

    Args:
        labels: Array of frame labels ('H', 'A', 'D', 'X').
        half_blocks: List of (half, first_frame, last_frame) tuples.
        min_live_spell_frames: Minimum frames required for a valid possession spell flip (default: 35).
        possessor_history: Optional array of possessor codes (0=Home, 1=Away) to keep in sync.

    Returns:
        Regularized labels array (modified in place).
    """
    n_frames = len(labels)
    team_code = {LABEL_H: HOME_CODE, LABEL_A: AWAY_CODE}

    for _, first_frame, last_frame in half_blocks:
        s = max(0, first_frame)
        e = min(n_frames, last_frame + 1)
        if s >= e:
            continue

        live_starts, live_ends = find_true_runs(labels[s:e] != LABEL_D)

        for l_start, l_end in zip(live_starts.tolist(), live_ends.tolist()):
            abs_start = s + l_start
            seg = labels[abs_start:s + l_end]
            if len(seg) < 3:
                continue

            # Run-length encode the live segment once, then merge at run level
            breaks = (np.flatnonzero(seg[1:] != seg[:-1]) + 1).tolist()
            run_starts = [0] + breaks
            run_ends = breaks + [len(seg)]
            run_labels = [seg[start] for start in run_starts]

            r = 1
            while r < len(run_labels) - 1:
                surrounding = run_labels[r - 1]
                is_micro_flip = (
                    run_ends[r] - run_starts[r] < min_live_spell_frames
                    and surrounding in team_code
                    and run_labels[r] in team_code
                    and run_labels[r] != surrounding
                    and run_labels[r + 1] == surrounding
                )
                if not is_micro_flip:
                    r += 1
                    continue

                seg[run_starts[r]:run_ends[r]] = surrounding
                if possessor_history is not None:
                    possessor_history[abs_start + run_starts[r]:abs_start + run_ends[r]] = team_code[surrounding]

                # Runs r - 1, r and r + 1 now form a single spell; re-check it against its
                # new neighbours (earlier runs are unaffected by the merge)
                run_ends[r - 1] = run_ends[r + 1]
                del run_labels[r:r + 2], run_starts[r:r + 2], run_ends[r:r + 2]
                r = max(1, r - 1)

    return labels


def run_possession_state_machine(
    n_frames: int,
    ball_xy: np.ndarray,
    is_ball_uncertain: np.ndarray,
    is_dead_ball: np.ndarray,
    player_xy: np.ndarray,
    team_of_track: np.ndarray,
    team_names: Tuple[str, str],
    smart_tagging_events: List[Dict[str, Any]],
    half_blocks: List[Tuple[int, int, int]],
    control_radius: float = 1.8,
    min_control_frames: int = 22,
    scramble_abstain_frames: int = 15,
    restart_search_frames: int = 50,
    min_live_spell_frames: int = 35,
    **_unused: Any,
) -> PossessionResult:
    """Execute temporal hysteresis possession state machine across all frames.

    Args:
        n_frames: Total video frames (199,319).
        ball_xy: Reconstructed ball coordinates in metres, shape (n_frames, 2).
        is_ball_uncertain: Uncertainty mask for ball position, shape (n_frames,).
        is_dead_ball: Dead ball mask, shape (n_frames,).
        player_xy: Player coordinates in metres, shape (n_frames, n_slots, 2).
        team_of_track: Slot to team mapping (0 = Home, 1 = Away, anything else is ignored).
        team_names: Tuple of (home_name, away_name).
        smart_tagging_events: Parsed Smart Tagging events.
        half_blocks: List of (half, first_frame, last_frame) tuples from games.py.
        control_radius: Proximity threshold in metres for controlled ball possession (default: 1.8 m).
        min_control_frames: Consecutive frames required for opponent to take possession (default: 22 frames).
        scramble_abstain_frames: Consecutive contested frames before emitting 'X' (default: 15 frames).
        restart_search_frames: Forward search window in frames for restart attribution (default: 50 frames).
        min_live_spell_frames: Minimum frames required for valid possession spell flip (default: 35 frames / 1.4 s).
        **_unused: Ignored; accepts legacy keyword arguments (ball_speed, player_speed, fps).

    Returns:
        PossessionResult containing final labels array, team distances, and possessor history.
        Every label is one of 'H', 'A', 'D', 'X'; frames outside the halves are 'D'.

    Raises:
        ValueError: If a per-frame input does not have n_frames rows.
    """
    for name, values in (
        ("ball_xy", ball_xy),
        ("is_ball_uncertain", is_ball_uncertain),
        ("is_dead_ball", is_dead_ball),
        ("player_xy", player_xy),
    ):
        if len(values) != n_frames:
            raise ValueError(f"{name} has {len(values)} rows, expected n_frames={n_frames}")

    labels = np.full(n_frames, LABEL_D, dtype=object)
    possessor_history = np.full(n_frames, -1, dtype=np.int8)

    min_dist_home, min_dist_away = compute_team_distances_to_ball(
        player_xy=player_xy,
        ball_xy=ball_xy,
        team_of_track=team_of_track,
        is_ball_uncertain=is_ball_uncertain,
    )

    # Process each half separately so possession state never crosses halftime
    for half, first_frame, last_frame in half_blocks:
        first_frame = max(0, first_frame)
        last_frame = min(n_frames - 1, last_frame)
        if first_frame > last_frame:
            continue

        current_possessor = identify_half_kickoff_team(
            half=half,
            kickoff_frame=first_frame,
            smart_tagging_events=smart_tagging_events,
            team_names=team_names,
            min_dist_home=min_dist_home,
            min_dist_away=min_dist_away,
        )

        possession_established = True
        control_streak = 0
        opp_evidence = 0
        scramble_streak = 0
        was_dead_previous = True

        for t in range(first_frame, last_frame + 1):
            # 1. Dead Ball override
            if is_dead_ball[t]:
                labels[t] = LABEL_D
                possessor_history[t] = current_possessor
                opp_evidence = 0
                scramble_streak = 0
                control_streak = 0
                was_dead_previous = True
                continue

            # 2. Restart transition: Dead Ball -> Live Play
            if was_dead_previous:
                was_dead_previous = False
                restart_team = attribute_restart_possession(
                    frame_idx=t,
                    smart_tagging_events=smart_tagging_events,
                    team_names=team_names,
                    min_dist_home=min_dist_home,
                    min_dist_away=min_dist_away,
                    is_ball_uncertain=is_ball_uncertain,
                    search_forward_frames=restart_search_frames,
                )
                if restart_team is not None:
                    current_possessor = restart_team
                    possession_established = True
                else:
                    # Unconfirmed restart
                    possession_established = False
                opp_evidence = 0

            d_home = min_dist_home[t]
            d_away = min_dist_away[t]
            opponent_team = AWAY_CODE if current_possessor == HOME_CODE else HOME_CODE
            d_possessor = d_home if current_possessor == HOME_CODE else d_away
            d_opponent = d_away if current_possessor == HOME_CODE else d_home

            # 3. Handle ball tracking blackout / uncertain frames
            if is_ball_uncertain[t]:
                if possession_established:
                    # Maintain established possession during occluded dribble to protect macroF1
                    labels[t] = LABEL_H if current_possessor == HOME_CODE else LABEL_A
                else:
                    # Unestablished restart or contest: emit 'X'
                    labels[t] = LABEL_X

                possessor_history[t] = current_possessor
                scramble_streak = 0
                # opp_evidence is frozen during blackout
                continue

            # 4. Ball is tracked: check for actively contested duel / scramble
            is_contested = (
                (d_home <= control_radius)
                and (d_away <= control_radius)
                and (abs(d_home - d_away) < _CONTESTED_MARGIN_M)
            )

            if is_contested:
                scramble_streak += 1
                if scramble_streak >= scramble_abstain_frames:
                    # Unresolved scramble: emit 'X' and mark possession unestablished
                    labels[t] = LABEL_X
                    possessor_history[t] = current_possessor
                    possession_established = False
                    opp_evidence = 0
                    continue
            else:
                scramble_streak = 0

            # 5. Pass-Preserving Possession Evidence (opp_evidence)
            # Reset only when possessing team re-establishes clear proximity
            if d_possessor <= _POSSESSOR_RESET_RADIUS_M and d_possessor < d_opponent:
                opp_evidence = 0
            # Continue accumulating when opponent receives / controls
            elif d_opponent <= _OPPONENT_EVIDENCE_RADIUS_M and d_opponent < d_possessor:
                opp_evidence += 1
                if opp_evidence >= min_control_frames:
                    # Turnover confirmed via sustained evidence
                    current_possessor = opponent_team
                    possession_established = True
                    opp_evidence = 0
                    control_streak = min_control_frames
            # Otherwise (free flight or ambiguous proximity): opp_evidence remains frozen

            # Confirm established possession if possessor stays near ball
            if d_possessor <= control_radius:
                control_streak += 1
                if control_streak >= _ESTABLISH_CONTROL_FRAMES:
                    possession_established = True
            else:
                control_streak = max(0, control_streak - 1)

            labels[t] = LABEL_H if current_possessor == HOME_CODE else LABEL_A
            possessor_history[t] = current_possessor

    # 6. Minimum live spell regularization (absorb micro-flips < min_live_spell_frames)
    regularize_live_spells(
        labels=labels,
        half_blocks=half_blocks,
        min_live_spell_frames=min_live_spell_frames,
        possessor_history=possessor_history,
    )

    # Enforce strictly valid labels
    invalid_mask = ~np.isin(labels, list(VALID_LABELS))
    if np.any(invalid_mask):
        labels[invalid_mask] = LABEL_D

    return PossessionResult(
        labels=labels,
        min_dist_home=min_dist_home,
        min_dist_away=min_dist_away,
        current_possessor=possessor_history,
    )
