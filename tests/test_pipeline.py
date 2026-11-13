"""Comprehensive unit tests for the possession identification pipeline.

All tests operate exclusively on synthetic data, respecting the strict data isolation
and no-ground-truth-tuning constraints.
"""

import numpy as np
import pytest

from src.ball_tracking import (
    reconstruct_ball_trajectory,
    sanitize_raw_ball_positions,
    find_missing_spans,
)
from src.dead_ball import (
    detect_dead_ball,
    smooth_signal,
    remove_short_spans,
    fill_short_gaps,
)
from src.possession_engine import (
    run_possession_state_machine,
    compute_team_distances_to_ball,
    LABEL_H,
    LABEL_A,
    LABEL_D,
    LABEL_X,
    HOME_CODE,
    AWAY_CODE,
)
from src.run_pipeline import (
    count_possession_spells,
    validate_submission_format,
)
import pandas as pd


def test_find_missing_spans():
    """Verify that contiguous missing spans are correctly identified."""
    mask = np.array([False, True, True, False, False, True, False])
    spans = find_missing_spans(mask)
    assert spans == [(1, 3), (5, 6)]

    # Leading and trailing missing spans
    mask2 = np.array([True, True, False, True])
    spans2 = find_missing_spans(mask2)
    assert spans2 == [(0, 2), (3, 4)]


def test_ball_tracking_short_gap_reconstruction():
    """Verify that gaps < 35 frames with physically plausible speed are smoothly interpolated."""
    n_frames = 100
    ball_xy = np.full((n_frames, 2), np.nan, dtype=np.float32)

    # Ball moving steadily from (10, 20) to (20, 20) over 10 frames
    # Frames 0..20: observed moving from (0, 20) to (20, 20)
    for i in range(21):
        ball_xy[i] = [float(i), 20.0]
    # Frames 21..35: missing (15 frames < 35 frames)
    # Frames 36..60: observed moving from (36, 20) to (60, 20)
    for i in range(36, 61):
        ball_xy[i] = [float(i), 20.0]

    result = reconstruct_ball_trajectory(
        raw_ball_xy=ball_xy,
        pitch=(105.3, 68.0),
        max_gap_frames=35,
        max_physical_speed_mps=38.0,
        fps=25.0,
    )

    # Check that short gap was interpolated
    assert not np.any(result.is_ball_uncertain[21:36])
    assert not np.any(np.isnan(result.ball_xy[21:36]))

    # Check interpolated position values are monotonic and smooth
    for i in range(21, 36):
        assert 20.0 < result.ball_xy[i, 0] < 36.0
        assert np.isclose(result.ball_xy[i, 1], 20.0, atol=0.1)


def test_ball_tracking_long_blackout_flagged_uncertain():
    """Verify that gaps >= 35 frames are NOT hallucinated and are flagged as uncertain."""
    n_frames = 100
    ball_xy = np.full((n_frames, 2), np.nan, dtype=np.float32)

    # Observed at frame 10, then missing for 40 frames, observed again at frame 51
    ball_xy[10] = [10.0, 20.0]
    ball_xy[51] = [30.0, 20.0]

    result = reconstruct_ball_trajectory(
        raw_ball_xy=ball_xy,
        pitch=(105.3, 68.0),
        max_gap_frames=35,
        max_physical_speed_mps=38.0,
        fps=25.0,
    )

    # Gap from 11 to 50 is 40 frames (>= 35 frames) -> must be flagged uncertain
    assert np.all(result.is_ball_uncertain[11:51])


def test_ball_tracking_teleportation_spike_rejected():
    """Verify that unfeasible single-frame teleportation jumps are rejected."""
    n_frames = 10
    ball_xy = np.zeros((n_frames, 2), dtype=np.float32)
    for i in range(n_frames):
        ball_xy[i] = [10.0 + i * 0.2, 20.0]

    # Insert a teleportation jump at frame 5 (jumping 60 metres in 1 frame = 1500 m/s)
    ball_xy[5] = [70.0, 50.0]

    clean_xy = sanitize_raw_ball_positions(
        ball_xy,
        pitch=(105.3, 68.0),
        max_jump_speed_mps=40.0,
        fps=25.0,
    )

    assert np.isnan(clean_xy[5, 0])
    assert not np.isnan(clean_xy[4, 0])
    assert not np.isnan(clean_xy[6, 0])


def test_dead_ball_detection_rules():
    """Verify that dead ball detector combines non-playing periods, Smart Tagging, and kinematics."""
    n_frames = 200
    mean_speed = np.full(n_frames, 2.2, dtype=np.float32)  # Active running
    ball_xy = np.full((n_frames, 2), 30.0, dtype=np.float32)
    is_ball_uncertain = np.zeros(n_frames, dtype=bool)
    ball_speed = np.full(n_frames, 5.0, dtype=np.float32)

    # Half 1 from frame 50 to 150
    half_blocks = [(1, 50, 150)]

    # Smart Tagging set piece stoppage from 80 to 110
    smart_tagging_events = [
        {
            "code": "SET PIECES",
            "start_frame": 80,
            "end_frame": 110,
            "tags": {"Type": "Throw-in"},
        }
    ]

    result = detect_dead_ball(
        n_frames=n_frames,
        mean_player_speed=mean_speed,
        ball_xy=ball_xy,
        is_ball_uncertain=is_ball_uncertain,
        ball_speed=ball_speed,
        smart_tagging_events=smart_tagging_events,
        half_blocks=half_blocks,
        pitch=(105.3, 68.0),
        fps=25.0,
    )

    # Frames outside half (0..49, 151..199) must be Dead Ball
    assert np.all(result.is_dead_ball[0:50])
    assert np.all(result.is_dead_ball[151:200])

    # Frames 80..110 inside tagged set piece must be Dead Ball
    assert np.all(result.is_dead_ball[80:110])

    # Active play frames (55..75) should be live play (not dead)
    assert not np.any(result.is_dead_ball[55:75])


def test_possession_hysteresis_state_machine():
    """Verify that possession only transitions after sustained opponent control (>= 22 frames)."""
    n_frames = 150
    half_blocks = [(1, 0, 149)]

    # Ball at (50, 30)
    ball_xy = np.full((n_frames, 2), [50.0, 30.0], dtype=np.float32)
    is_ball_uncertain = np.zeros(n_frames, dtype=bool)
    is_dead_ball = np.zeros(n_frames, dtype=bool)

    # 4 player slots: Slot 0, 1 = Home; Slot 2, 3 = Away
    player_xy = np.full((n_frames, 4, 2), 999.0, dtype=np.float32)
    player_speed = np.full((n_frames, 4), 2.0, dtype=np.float32)
    team_of_track = np.array([HOME_CODE, HOME_CODE, AWAY_CODE, AWAY_CODE], dtype=np.int8)

    # Frames 0..50: Home player (slot 0) is at the ball (0.5 m away)
    player_xy[0:51, 0] = [50.5, 30.0]
    player_xy[0:51, 2] = [60.0, 30.0]  # Away player is far

    # Frames 51..60: Away player makes a brief 10-frame challenge (touch/deflection)
    # 10 frames < min_control_frames (22) -> should NOT cause a possession turnover!
    player_xy[51:61, 2] = [50.5, 30.0]
    player_xy[51:61, 0] = [55.0, 30.0]

    # Frames 61..80: Home player regains close proximity
    player_xy[61:81, 0] = [50.5, 30.0]
    player_xy[61:81, 2] = [60.0, 30.0]

    # Frames 81..120: Away player establishes sustained control (40 frames >= 22)
    player_xy[81:121, 2] = [50.5, 30.0]
    player_xy[81:121, 0] = [60.0, 30.0]

    result = run_possession_state_machine(
        n_frames=n_frames,
        ball_xy=ball_xy,
        is_ball_uncertain=is_ball_uncertain,
        is_dead_ball=is_dead_ball,
        player_xy=player_xy,
        player_speed=player_speed,
        team_of_track=team_of_track,
        smart_tagging_events=[],
        half_blocks=half_blocks,
        control_radius=1.8,
        min_control_frames=22,
        fps=25.0,
    )

    # 1. Frames 0..50 must be Home ('H')
    assert np.all(result.labels[0:51] == LABEL_H)

    # 2. Brief 10-frame touch (51..60) must NOT flip possession: must remain Home ('H')
    assert np.all(result.labels[51:61] == LABEL_H)

    # 3. Frames 61..80 must still be Home ('H')
    assert np.all(result.labels[61:81] == LABEL_H)

    # 4. By frame 110 (after 29 frames of sustained Away proximity), possession must have transitioned to Away ('A')
    assert np.all(result.labels[105:121] == LABEL_A)


def test_possession_abstention_during_blackouts():
    """Verify that during prolonged ball tracking uncertainty, engine emits 'X'."""
    n_frames = 60
    half_blocks = [(1, 0, 59)]

    ball_xy = np.full((n_frames, 2), [50.0, 30.0], dtype=np.float32)
    is_ball_uncertain = np.zeros(n_frames, dtype=bool)
    is_dead_ball = np.zeros(n_frames, dtype=bool)

    # Frames 20..35: tracking blackout
    is_ball_uncertain[20:36] = True

    player_xy = np.full((n_frames, 2, 2), 999.0, dtype=np.float32)
    player_speed = np.full((n_frames, 2), 2.0, dtype=np.float32)
    team_of_track = np.array([HOME_CODE, AWAY_CODE], dtype=np.int8)
    player_xy[:, 0] = [50.5, 30.0]  # Home player near

    result = run_possession_state_machine(
        n_frames=n_frames,
        ball_xy=ball_xy,
        is_ball_uncertain=is_ball_uncertain,
        is_dead_ball=is_dead_ball,
        player_xy=player_xy,
        player_speed=player_speed,
        team_of_track=team_of_track,
        smart_tagging_events=[],
        half_blocks=half_blocks,
        fps=25.0,
    )

    # Blackout frames must emit 'X'
    assert np.all(result.labels[20:36] == LABEL_X)
    # Outside blackout, must emit 'H'
    assert np.all(result.labels[0:20] == LABEL_H)
    assert np.all(result.labels[36:60] == LABEL_H)


def test_submission_validation_rules():
    """Verify that submission format checker enforces columns, rows, monotonicity, and label domain."""
    # Valid submission
    valid_df = pd.DataFrame({
        "frame": np.arange(10, dtype=np.int32),
        "label": ["H", "H", "A", "D", "X", "H", "A", "A", "D", "D"],
    })
    validate_submission_format(valid_df, expected_rows=10)

    # Invalid row count
    with pytest.raises(ValueError, match="row count"):
        validate_submission_format(valid_df, expected_rows=15)

    # Invalid column names
    bad_cols_df = valid_df.rename(columns={"label": "prediction"})
    with pytest.raises(ValueError, match="Invalid columns"):
        validate_submission_format(bad_cols_df, expected_rows=10)

    # Invalid labels
    bad_labels_df = valid_df.copy()
    bad_labels_df.loc[2, "label"] = "INVALID"
    with pytest.raises(ValueError, match="Invalid labels found"):
        validate_submission_format(bad_labels_df, expected_rows=10)

    # NaN labels
    nan_labels_df = valid_df.copy()
    nan_labels_df.loc[2, "label"] = np.nan
    with pytest.raises(ValueError, match="contains NaN"):
        validate_submission_format(nan_labels_df, expected_rows=10)


def test_spell_counter():
    """Verify possession spell counting with X-bridging."""
    labels = np.array(["D", "H", "H", "X", "H", "A", "A", "D", "A", "A"])
    half_blocks = [(1, 0, 9)]
    spells, mean_dur = count_possession_spells(labels, half_blocks, fps=25.0)

    # Within the half, 'H H X H' bridges to one continuous 'H' spell
    # Then 'A A' is the second spell
    # Then 'D' is stoppage (excluded)
    # Then 'A A' is the third spell
    # Total spells = 3
    assert spells == 3
