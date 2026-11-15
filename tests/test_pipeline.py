"""Comprehensive unit tests for the refactored possession identification pipeline.

All tests operate exclusively on synthetic data, respecting the strict data isolation
and no-ground-truth-tuning constraints.
"""

import numpy as np
import pytest
import pandas as pd

from src.config import PipelineConfig
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
    match_team_string,
    regularize_live_spells,
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


def test_find_missing_spans():
    """Verify that contiguous missing spans are correctly identified."""
    mask = np.array([False, True, True, False, False, True, False])
    spans = find_missing_spans(mask)
    assert spans == [(1, 3), (5, 6)]


def test_ball_tracking_short_gap_reconstruction():
    """Verify that short gaps (< 35 frames) with feasible speeds are smoothly interpolated."""
    n_frames = 100
    ball_xy = np.full((n_frames, 2), np.nan, dtype=np.float32)

    for i in range(21):
        ball_xy[i] = [float(i), 20.0]
    for i in range(36, 61):
        ball_xy[i] = [float(i), 20.0]

    result = reconstruct_ball_trajectory(
        raw_ball_xy=ball_xy,
        pitch=(105.3, 68.0),
        max_gap_frames=35,
        max_physical_speed_mps=38.0,
        fps=25.0,
    )

    assert not np.any(result.is_ball_uncertain[21:36])
    assert not np.any(np.isnan(result.ball_xy[21:36]))
    # Velocity should be non-NaN and physically reasonable (~25 m/s)
    assert np.all(result.ball_speed[21:36] > 10.0)
    assert np.all(result.ball_speed[21:36] < 35.0)


def test_ball_tracking_multiframe_spike_rejected():
    """Verify that 2-3 frame false-positive spikes are eliminated via bidirectional reachability."""
    n_frames = 20
    ball_xy = np.zeros((n_frames, 2), dtype=np.float32)
    for i in range(n_frames):
        ball_xy[i] = [10.0 + i * 0.2, 20.0]

    # Insert a 2-frame spike at frames 5 and 6 (teleporting 40m away)
    ball_xy[5] = [50.0, 50.0]
    ball_xy[6] = [51.0, 50.0]

    clean_xy = sanitize_raw_ball_positions(
        ball_xy,
        pitch=(105.3, 68.0),
        max_jump_speed_mps=40.0,
        fps=25.0,
    )

    assert np.isnan(clean_xy[5, 0])
    assert np.isnan(clean_xy[6, 0])
    assert not np.isnan(clean_xy[4, 0])
    assert not np.isnan(clean_xy[7, 0])


def test_match_team_string():
    """Verify dynamic team string matching against metadata names."""
    home_name = "FC Barcelona"
    away_name = "Athletic Club"

    assert match_team_string("Barcelona", home_name, away_name) == HOME_CODE
    assert match_team_string("Barca", home_name, away_name) == HOME_CODE
    assert match_team_string("Atletic Club", home_name, away_name) == AWAY_CODE
    assert match_team_string("Athletic", home_name, away_name) == AWAY_CODE
    assert match_team_string("N/A", home_name, away_name) is None


def test_dead_ball_p80_kinematics():
    """Verify that kinematic D triggers only when p80 outfield speed is sustained low in stoppage context."""
    n_frames = 200
    mean_speed = np.full(n_frames, 1.8, dtype=np.float32)
    p80_speed = np.full(n_frames, 2.2, dtype=np.float32)
    ball_xy = np.full((n_frames, 2), [50.0, 30.0], dtype=np.float32)
    is_ball_uncertain = np.zeros(n_frames, dtype=bool)
    ball_speed = np.full(n_frames, 4.0, dtype=np.float32)
    ball_speed[50:70] = 0.0  # Stationary ball during set piece setup
    player_xy = np.full((n_frames, 4, 2), [50.0, 30.0], dtype=np.float32)

    half_blocks = [(1, 0, 199)]

    # Set piece stoppage from 50 to 70
    smart_tagging_events = [
        {"code": "SET PIECES", "start_frame": 50, "end_frame": 70, "tags": {"Type": "Throw-in"}}
    ]

    # In frames 65 to 130 (65 frames >= 50), p80 speed drops to 1.1 m/s (walking) near the stoppage
    p80_speed[65:135] = 1.1
    mean_speed[65:135] = 0.9

    result = detect_dead_ball(
        n_frames=n_frames,
        mean_player_speed=mean_speed,
        p80_player_speed=p80_speed,
        ball_xy=ball_xy,
        is_ball_uncertain=is_ball_uncertain,
        ball_speed=ball_speed,
        player_xy=player_xy,
        smart_tagging_events=smart_tagging_events,
        half_blocks=half_blocks,
        fps=25.0,
    )

    # Tagged set piece (50..70) is dead
    assert np.all(result.is_dead_ball[50:70])
    # Sustained low p80 speed within dilated window (70..120) should also be dead
    assert np.all(result.is_dead_ball[70:120])
    # Active open play before frame 45 is alive
    assert not np.any(result.is_dead_ball[0:45])


def test_dead_ball_out_of_bounds_latching():
    """Verify that out-of-bounds latches is_dead forward across uncertain frames until restart."""
    n_frames = 100
    half_blocks = [(1, 0, 99)]
    mean_speed = np.full(n_frames, 2.5, dtype=np.float32)
    p80_speed = np.full(n_frames, 3.0, dtype=np.float32)
    ball_xy = np.full((n_frames, 2), [50.0, 30.0], dtype=np.float32)
    is_ball_uncertain = np.zeros(n_frames, dtype=bool)
    ball_speed = np.full(n_frames, 5.0, dtype=np.float32)
    player_xy = np.full((n_frames, 2, 2), 999.0, dtype=np.float32)

    # At frame 20, ball exits over touchline (y = 70.0 m > 68.0 + 0.6 m)
    ball_xy[20] = [50.0, 70.0]
    # Frames 21..50: ball tracking uncertain (waiting for throw-in)
    is_ball_uncertain[21:51] = True
    # At frame 51: ball re-enters pitch (x=50.0, y=34.0) with player at (50.5, 34.0) (dist 0.5m <= 2.2m)
    ball_xy[51:] = [50.0, 34.0]
    is_ball_uncertain[51:] = False
    player_xy[51:, 0] = [50.5, 34.0]

    result = detect_dead_ball(
        n_frames=n_frames,
        mean_player_speed=mean_speed,
        p80_player_speed=p80_speed,
        ball_xy=ball_xy,
        is_ball_uncertain=is_ball_uncertain,
        ball_speed=ball_speed,
        player_xy=player_xy,
        smart_tagging_events=[],
        half_blocks=half_blocks,
        fps=25.0,
    )

    # Out-of-bounds at frame 20 and latched across uncertain frames 21..50 should all be Dead
    assert np.all(result.is_dead_ball[20:51])
    # Once re-entered in controlled play (frame 51 onwards), play is live
    assert not np.any(result.is_dead_ball[51:80])


def test_standalone_collective_stoppage():
    """Verify that p80 outfield speed < 1.35 m/s sustained for >= 60 frames marks D outside Smart Tagging."""
    n_frames = 150
    half_blocks = [(1, 0, 149)]
    mean_speed = np.full(n_frames, 2.5, dtype=np.float32)
    p80_speed = np.full(n_frames, 3.0, dtype=np.float32)
    ball_xy = np.full((n_frames, 2), [50.0, 30.0], dtype=np.float32)
    is_ball_uncertain = np.zeros(n_frames, dtype=bool)
    ball_speed = np.full(n_frames, 5.0, dtype=np.float32)
    player_xy = np.full((n_frames, 2, 2), [50.0, 30.0], dtype=np.float32)

    # Standalone collective drop: p80 drops to 1.2 m/s for 70 frames (frames 30..100)
    p80_speed[30:101] = 1.2
    mean_speed[30:101] = 1.0

    result = detect_dead_ball(
        n_frames=n_frames,
        mean_player_speed=mean_speed,
        p80_player_speed=p80_speed,
        ball_xy=ball_xy,
        is_ball_uncertain=is_ball_uncertain,
        ball_speed=ball_speed,
        player_xy=player_xy,
        smart_tagging_events=[],  # No Smart Tagging
        half_blocks=half_blocks,
        standalone_stoppage_frames=60,
        standalone_p80_threshold_mps=1.35,
        fps=25.0,
    )

    # Should mark dead ball across sustained low speed window
    assert np.all(result.is_dead_ball[35:95])
    # Live play before frame 25
    assert not np.any(result.is_dead_ball[0:25])


def test_pass_preserving_possession_evidence():
    """Verify that a fast pass flying past a pressing opponent does NOT flip possession."""
    n_frames = 60
    half_blocks = [(1, 0, 59)]

    ball_xy = np.full((n_frames, 2), [50.0, 30.0], dtype=np.float32)
    is_ball_uncertain = np.zeros(n_frames, dtype=bool)
    is_dead_ball = np.zeros(n_frames, dtype=bool)
    ball_speed = np.full(n_frames, 12.0, dtype=np.float32)

    player_xy = np.full((n_frames, 2, 2), 999.0, dtype=np.float32)
    player_speed = np.full((n_frames, 2), 2.0, dtype=np.float32)
    team_of_track = np.array([HOME_CODE, AWAY_CODE], dtype=np.int8)

    # Home possesses initially (frames 0..15)
    player_xy[0:16, 0] = [50.5, 30.0]

    # Frames 16..18 (3 frames): pass flies past an Away player (dist = 1.5 m <= 2.0 m)
    player_xy[16:19, 1] = [51.5, 30.0]

    # Frames 19..25: ball continues in free flight to Home teammate (both > 2.2 m)
    # (player_xy remains 999.0)

    # Frames 26..50: Home teammate controls pass (dist = 1.0 m <= 1.8 m)
    player_xy[26:51, 0] = [51.0, 30.0]

    result = run_possession_state_machine(
        n_frames=n_frames,
        ball_xy=ball_xy,
        is_ball_uncertain=is_ball_uncertain,
        is_dead_ball=is_dead_ball,
        ball_speed=ball_speed,
        player_xy=player_xy,
        player_speed=player_speed,
        team_of_track=team_of_track,
        team_names=("FC Barcelona", "Athletic Club"),
        smart_tagging_events=[],
        half_blocks=half_blocks,
        fps=25.0,
    )

    # Possession should NEVER have flipped to Away ('A')
    assert np.all(result.labels[0:51] == LABEL_H)


def test_possession_turnover_after_sustained_control():
    """Verify that opponent takes possession after sustained control (22 frames)."""
    n_frames = 60
    half_blocks = [(1, 0, 59)]

    ball_xy = np.full((n_frames, 2), [50.0, 30.0], dtype=np.float32)
    is_ball_uncertain = np.zeros(n_frames, dtype=bool)
    is_dead_ball = np.zeros(n_frames, dtype=bool)
    ball_speed = np.full(n_frames, 2.0, dtype=np.float32)

    player_xy = np.full((n_frames, 2, 2), 999.0, dtype=np.float32)
    player_speed = np.full((n_frames, 2), 2.0, dtype=np.float32)
    team_of_track = np.array([HOME_CODE, AWAY_CODE], dtype=np.int8)

    # Home controls 0..15
    player_xy[0:16, 0] = [50.5, 30.0]

    # Away player takes ball at frame 16 and holds it for 25 frames (16..40)
    player_xy[16:41, 1] = [50.5, 30.0]

    result = run_possession_state_machine(
        n_frames=n_frames,
        ball_xy=ball_xy,
        is_ball_uncertain=is_ball_uncertain,
        is_dead_ball=is_dead_ball,
        ball_speed=ball_speed,
        player_xy=player_xy,
        player_speed=player_speed,
        team_of_track=team_of_track,
        team_names=("FC Barcelona", "Athletic Club"),
        smart_tagging_events=[],
        half_blocks=half_blocks,
        min_control_frames=22,
        fps=25.0,
    )

    # Frames 0..15 are Home ('H')
    assert np.all(result.labels[0:16] == LABEL_H)
    # After 22 frames of Away control (frame 16 + 22 = 38 onwards), should be Away ('A')
    assert np.all(result.labels[38:41] == LABEL_A)


def test_minimum_live_spell_regularization():
    """Verify that brief micro-flips (< 35 frames) are absorbed into the surrounding team's spell."""
    labels = np.array(
        [LABEL_H] * 50 + [LABEL_A] * 20 + [LABEL_H] * 50,
        dtype=object,
    )
    half_blocks = [(1, 0, len(labels) - 1)]

    regularized = regularize_live_spells(labels, half_blocks, min_live_spell_frames=35)
    # The 20-frame 'A' micro-flip surrounded by 'H' must be absorbed into 'H'
    assert np.all(regularized == LABEL_H)


def test_possession_selective_abstention():
    """Verify that established possession is maintained during occlusions, while contested scrambles emit 'X'."""
    n_frames = 80
    half_blocks = [(1, 0, 79)]

    ball_xy = np.full((n_frames, 2), [50.0, 30.0], dtype=np.float32)
    is_ball_uncertain = np.zeros(n_frames, dtype=bool)
    is_dead_ball = np.zeros(n_frames, dtype=bool)
    ball_speed = np.full(n_frames, 1.5, dtype=np.float32)

    player_xy = np.full((n_frames, 2, 2), 999.0, dtype=np.float32)
    player_speed = np.full((n_frames, 2), 2.0, dtype=np.float32)
    team_of_track = np.array([HOME_CODE, AWAY_CODE], dtype=np.int8)

    # Home firmly controls from 0..25 (possession established)
    player_xy[0:26, 0] = [50.5, 30.0]

    # Frames 26..35: ball tracking blackout during established Home dribble
    is_ball_uncertain[26:36] = True

    # Frames 40..65: contested 50/50 scramble between Home and Away
    player_xy[40:66, 0] = [50.5, 30.0]
    player_xy[40:66, 1] = [50.6, 30.0]

    result = run_possession_state_machine(
        n_frames=n_frames,
        ball_xy=ball_xy,
        is_ball_uncertain=is_ball_uncertain,
        is_dead_ball=is_dead_ball,
        ball_speed=ball_speed,
        player_xy=player_xy,
        player_speed=player_speed,
        team_of_track=team_of_track,
        team_names=("FC Barcelona", "Athletic Club"),
        smart_tagging_events=[],
        half_blocks=half_blocks,
        scramble_abstain_frames=15,
        fps=25.0,
    )

    # Established dribble during blackout (26..35) should maintain Home ('H')
    assert np.all(result.labels[26:36] == LABEL_H)

    # Contested scramble (after 15 frames, i.e. 55..65) must emit 'X'
    assert np.all(result.labels[56:66] == LABEL_X)


def test_submission_validation_rules():
    """Verify that submission format checker enforces columns, rows, monotonicity, and label domain."""
    valid_df = pd.DataFrame({
        "frame": np.arange(10, dtype=np.int32),
        "label": ["H", "H", "A", "D", "X", "H", "A", "A", "D", "D"],
    })
    validate_submission_format(valid_df, expected_rows=10)

    # Invalid row count
    with pytest.raises(ValueError, match="row count"):
        validate_submission_format(valid_df, expected_rows=15)

    # Invalid columns
    bad_cols_df = valid_df.rename(columns={"label": "prediction"})
    with pytest.raises(ValueError, match="Invalid columns"):
        validate_submission_format(bad_cols_df, expected_rows=10)

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

    # Within the half: 'H H X H' bridges to one continuous 'H' spell
    # Then 'A A' is the second spell
    # Then 'D' is stoppage
    # Then 'A A' is the third spell -> Total = 3
    assert spells == 3


def test_vectorized_speed_statistics():
    """Verify that vectorized outfield speed stats compute identical values to row-wise math without errors."""
    n_frames = 50
    n_slots = 22
    outfield_indices = np.arange(20)  # slots 0..19 outfield, slots 20, 21 GKs
    player_speed = np.random.uniform(0.5, 6.0, size=(n_frames, n_slots)).astype(np.float32)
    # Add some NaNs
    player_speed[10, :5] = np.nan
    player_speed[20, :] = np.nan

    outfield_speeds = player_speed[:, outfield_indices]
    with np.errstate(invalid="ignore"):
        mean_speed = np.nanmean(outfield_speeds, axis=1)
        p80_speed = np.nanpercentile(outfield_speeds, 80, axis=1)
    mean_speed = np.nan_to_num(mean_speed, nan=0.0).astype(np.float32)
    p80_speed = np.nan_to_num(p80_speed, nan=0.0).astype(np.float32)

    assert len(mean_speed) == n_frames
    assert len(p80_speed) == n_frames
    assert mean_speed[20] == 0.0
    assert p80_speed[20] == 0.0
    # Frame 0 should match np.mean
    assert np.isclose(mean_speed[0], np.mean(player_speed[0, :20]))
    assert np.isclose(p80_speed[0], np.percentile(player_speed[0, :20], 80))

