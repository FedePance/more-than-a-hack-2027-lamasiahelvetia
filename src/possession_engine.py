"""Hysteresis possession state machine for robust possession spell identification.

Combines:
1. Dynamic team name resolution matching EPTS tracking metadata against Smart Tagging tags.
2. Smart restart attribution prioritizing Smart Tagging 'Team' tags and 50-frame forward search.
3. Pass-preserving possession evidence accumulation (opp_evidence) with free-flight freezing.
4. Minimum live spell regularization absorbing brief micro-flips (< 35 frames) into the surrounding spell.
5. Selective abstention ('X') during unresolved contested scrambles or unestablished restart blackouts,
   while maintaining established possession during occluded dribbles.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional
import numpy as np


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
    clean_home = home_name.lower().replace(" ", "").replace("-", "")
    clean_away = away_name.lower().replace(" ", "").replace("-", "")

    # Exact or substring containment
    if clean_tag in clean_home or clean_home in clean_tag:
        return HOME_CODE
    if clean_tag in clean_away or clean_away in clean_tag:
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
    """Compute per-frame minimum Euclidean distance from each team to the ball in metres."""
    n_frames = len(ball_xy)
    home_slots = np.flatnonzero(team_of_track == HOME_CODE)
    away_slots = np.flatnonzero(team_of_track == AWAY_CODE)

    min_dist_home = np.full(n_frames, 999.0, dtype=np.float32)
    min_dist_away = np.full(n_frames, 999.0, dtype=np.float32)

    known_ball = ~is_ball_uncertain & ~np.isnan(ball_xy[:, 0])
    valid_frames = np.flatnonzero(known_ball)

    if not len(valid_frames):
        return min_dist_home, min_dist_away

    bx = ball_xy[valid_frames, 0:1]
    by = ball_xy[valid_frames, 1:2]

    # Home players
    if len(home_slots):
        hx = player_xy[valid_frames][:, home_slots, 0]
        hy = player_xy[valid_frames][:, home_slots, 1]
        dx_h = hx - bx
        dy_h = hy - by
        dist_sq_h = dx_h * dx_h + dy_h * dy_h
        with np.errstate(invalid="ignore"):
            min_sq_h = np.nanmin(dist_sq_h, axis=1)
        min_sq_h = np.nan_to_num(min_sq_h, nan=999.0 * 999.0)
        min_dist_home[valid_frames] = np.sqrt(np.maximum(0.0, min_sq_h)).astype(np.float32)

    # Away players
    if len(away_slots):
        ax = player_xy[valid_frames][:, away_slots, 0]
        ay = player_xy[valid_frames][:, away_slots, 1]
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
            event_type = (event.get("tags", {}).get("Type") or "").strip().lower()
            event_half = (event.get("tags", {}).get("Half") or "").strip().lower()
            if "kick off" in event_type and (event_half == half_str.lower() or event_half == str(half)):
                tag_team = event.get("tags", {}).get("Team") or ""
                matched = match_team_string(tag_team, home_name, away_name)
                if matched is not None:
                    return matched

    # Physical fallback: player closest to ball at kickoff frame
    check_window = 5
    start = max(0, kickoff_frame)
    end = min(len(min_dist_home), kickoff_frame + check_window)
    h_dist = np.min(min_dist_home[start:end]) if end > start else 999.0
    a_dist = np.min(min_dist_away[start:end]) if end > start else 999.0

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

    1. First checks the preceding or overlapping Smart Tagging set-piece event Team tag.
    2. Physical fallback: searches forward up to search_forward_frames (2.0 s) for the first
       frame where a player is close to the ball (<= 2.5 m) to identify the restart taker.
    """
    home_name, away_name = team_names

    # Check Smart Tagging set-piece event near this restart
    for event in smart_tagging_events:
        start_f = event.get("start_frame", 0)
        end_f = event.get("end_frame", 0)
        # Event overlaps or precedes this restart frame within 4 seconds
        if start_f - 50 <= frame_idx <= end_f + 50:
            tag_team = event.get("tags", {}).get("Team") or ""
            matched = match_team_string(tag_team, home_name, away_name)
            if matched is not None:
                return matched

    # Physical fallback: inspect the first frame(s) at restart where ball is visible
    n_frames = len(min_dist_home)
    search_end = min(n_frames, frame_idx + search_forward_frames)

    for f in range(frame_idx, search_end):
        if not is_ball_uncertain[f]:
            h_d = min_dist_home[f]
            a_d = min_dist_away[f]
            if h_d <= 2.5 or a_d <= 2.5:
                return HOME_CODE if h_d <= a_d else AWAY_CODE

    return None


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
        Regularized labels array.
    """
    n_frames = len(labels)
    for _, first_frame, last_frame in half_blocks:
        s = max(0, first_frame)
        e = min(n_frames, last_frame + 1)
        if s >= e:
            continue

        half_labels = labels[s:e]
        is_live = half_labels != LABEL_D

        diffs = np.diff(is_live.astype(np.int8))
        live_starts = (np.flatnonzero(diffs == 1) + 1).tolist()
        live_ends = (np.flatnonzero(diffs == -1) + 1).tolist()
        if is_live[0]:
            live_starts = [0] + live_starts
        if is_live[-1]:
            live_ends = live_ends + [len(is_live)]

        for l_start, l_end in zip(live_starts, live_ends):
            abs_start = s + l_start
            abs_end = s + l_end
            seg = labels[abs_start:abs_end]

            while True:
                if len(seg) < 3:
                    break

                run_labels = []
                run_starts = []
                run_ends = []

                cur_lbl = seg[0]
                cur_s = 0
                for idx in range(1, len(seg)):
                    if seg[idx] != cur_lbl:
                        run_labels.append(cur_lbl)
                        run_starts.append(cur_s)
                        run_ends.append(idx)
                        cur_lbl = seg[idx]
                        cur_s = idx
                run_labels.append(cur_lbl)
                run_starts.append(cur_s)
                run_ends.append(len(seg))

                absorbed = False
                for r in range(1, len(run_labels) - 1):
                    prev_l = run_labels[r - 1]
                    cur_l = run_labels[r]
                    next_l = run_labels[r + 1]
                    run_len = run_ends[r] - run_starts[r]

                    if run_len < min_live_spell_frames:
                        if cur_l == LABEL_A and prev_l == LABEL_H and next_l == LABEL_H:
                            seg[run_starts[r]:run_ends[r]] = LABEL_H
                            if possessor_history is not None:
                                possessor_history[abs_start + run_starts[r]:abs_start + run_ends[r]] = HOME_CODE
                            absorbed = True
                            break
                        elif cur_l == LABEL_H and prev_l == LABEL_A and next_l == LABEL_A:
                            seg[run_starts[r]:run_ends[r]] = LABEL_A
                            if possessor_history is not None:
                                possessor_history[abs_start + run_starts[r]:abs_start + run_ends[r]] = AWAY_CODE
                            absorbed = True
                            break

                if not absorbed:
                    break

    return labels


def run_possession_state_machine(
    n_frames: int,
    ball_xy: np.ndarray,
    is_ball_uncertain: np.ndarray,
    is_dead_ball: np.ndarray,
    ball_speed: np.ndarray,
    player_xy: np.ndarray,
    player_speed: np.ndarray,
    team_of_track: np.ndarray,
    team_names: Tuple[str, str],
    smart_tagging_events: List[Dict[str, Any]],
    half_blocks: List[Tuple[int, int, int]],
    control_radius: float = 1.8,
    min_control_frames: int = 22,
    scramble_abstain_frames: int = 15,
    restart_search_frames: int = 50,
    min_live_spell_frames: int = 35,
    fps: float = 25.0,
    **kwargs,
) -> PossessionResult:
    """Execute temporal hysteresis possession state machine across all frames.

    Args:
        n_frames: Total video frames (199,319).
        ball_xy: Reconstructed ball coordinates in metres, shape (n_frames, 2).
        is_ball_uncertain: Uncertainty mask for ball position, shape (n_frames,).
        is_dead_ball: Dead ball mask, shape (n_frames,).
        ball_speed: Estimated ball speed in m/s, shape (n_frames,).
        player_xy: Player coordinates in metres, shape (n_frames, n_slots, 2).
        player_speed: Player speeds in m/s, shape (n_frames, n_slots).
        team_of_track: Slot to team mapping (0 = Home, 1 = Away).
        team_names: Tuple of (home_name, away_name).
        smart_tagging_events: Parsed Smart Tagging events.
        half_blocks: List of (half, first_frame, last_frame) tuples from games.py.
        control_radius: Proximity threshold in metres for controlled ball possession (default: 1.8 m).
        min_control_frames: Consecutive frames required for opponent to take possession (default: 22 frames).
        scramble_abstain_frames: Consecutive contested frames before emitting 'X' (default: 15 frames).
        restart_search_frames: Forward search window in frames for restart attribution (default: 50 frames).
        min_live_spell_frames: Minimum frames required for valid possession spell flip (default: 35 frames / 1.4 s).
        fps: Video frame rate.

    Returns:
        PossessionResult containing final labels array, team distances, and possessor history.
    """
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
                and (abs(d_home - d_away) < 0.4)
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
            if d_possessor <= 1.8 and d_possessor < d_opponent:
                opp_evidence = 0
            # Continue accumulating when opponent receives / controls
            elif d_opponent <= 2.0 and d_opponent < d_possessor:
                opp_evidence += 1
                if opp_evidence >= min_control_frames:
                    # Turnover confirmed via sustained evidence
                    current_possessor = opponent_team
                    possession_established = True
                    opp_evidence = 0
                    control_streak = min_control_frames
            elif d_home > 2.2 and d_away > 2.2:
                # Free flight: freeze opp_evidence
                pass
            # Otherwise (neither clear possessor proximity nor opponent receiving nor free flight):
            # opp_evidence remains frozen

            # Confirm established possession if possessor stays near ball
            if d_possessor <= control_radius:
                control_streak += 1
                if control_streak >= 15:
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
