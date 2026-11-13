"""Main CLI entrypoint for the 'More Than A Hack 2027' possession identification pipeline.

Team: La MasIA Helvetia
Challenge: Phase 1 (Ball Possession Identification)
Match: FC Barcelona vs Athletic Club (BAR-ATH)

Executes end-to-end:
1. Asset resolution via games.py and MTAH_DATA_DIR (fallback to ./data)
2. Streaming raw tracking and Smart Tagging loading
3. Ball trajectory reconstruction and ambiguity flagging (src/ball_tracking.py)
4. Dead ball detection combining Smart Tagging and player kinematics (src/dead_ball.py)
5. Temporal hysteresis possession state machine (src/possession_engine.py)
6. Submission export and verification against official format rules
"""

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Add repository root to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from games import (
    GAME,
    GAMES,
    DEFAULT_SOURCE,
    asset_paths,
    half_video_blocks,
    require,
    source_frames,
)
from starter.load_smart_tagging import load as load_smart_tagging
from src.data_loader import load_raw_tracking_data
from src.ball_tracking import reconstruct_ball_trajectory
from src.dead_ball import detect_dead_ball
from src.possession_engine import run_possession_state_machine


def resolve_data_directory(cli_data_dir: str = None) -> Path:
    """Resolve match data directory with precedence: CLI flag > MTAH_DATA_DIR > ./data.

    Args:
        cli_data_dir: Optional path provided via command line.

    Returns:
        Path object pointing to the resolved data directory.
    """
    if cli_data_dir:
        return Path(cli_data_dir).resolve()

    env_dir = os.environ.get("MTAH_DATA_DIR")
    if env_dir and env_dir.strip():
        return Path(env_dir.strip()).resolve()

    return (ROOT_DIR / "data").resolve()


def count_possession_spells(labels: np.ndarray, half_blocks: List[Tuple[int, int, int]], fps: float = 25.0) -> Tuple[int, float]:
    """Calculate the number and mean duration of live possession spells (maximal runs of H or A).

    Excludes dead ball ('D') and bridges abstentions ('X') within each half,
    matching the official scorer's spellF1 definition.

    Args:
        labels: Array of predicted labels ('H', 'A', 'D', 'X').
        half_blocks: List of (half, first_frame, last_frame) tuples.
        fps: Video frame rate.

    Returns:
        Tuple of (total_spells, mean_duration_seconds).
    """
    total_spells = 0
    total_live_frames = 0

    for _, start_frame, end_frame in half_blocks:
        s = max(0, start_frame)
        e = min(len(labels), end_frame + 1)
        window = labels[s:e].copy()

        # Forward fill and backfill X within half as the official scorer does
        known_indices = np.flatnonzero(window != "X")
        if not len(known_indices):
            continue

        carry = np.maximum.accumulate(np.where(window != "X", np.arange(len(window)), -1))
        carry[carry < 0] = known_indices[0]
        filled_window = window[carry]

        # Find continuous segments of identical labels
        breaks = np.flatnonzero(filled_window[1:] != filled_window[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [len(filled_window)]))

        for i in range(len(bounds) - 1):
            seg_start = bounds[i]
            seg_end = bounds[i + 1]
            seg_label = filled_window[seg_start]
            if seg_label in ("H", "A"):
                total_spells += 1
                total_live_frames += (seg_end - seg_start)

    mean_duration = (total_live_frames / fps / total_spells) if total_spells > 0 else 0.0
    return total_spells, mean_duration


def validate_submission_format(df: pd.DataFrame, expected_rows: int) -> None:
    """Verify that the generated DataFrame matches official submission specifications.

    Args:
        df: DataFrame containing 'frame' and 'label' columns.
        expected_rows: Expected total row count (e.g. 199,319).

    Raises:
        ValueError if the submission format does not comply with official rules.
    """
    if len(df) != expected_rows:
        raise ValueError(f"Submission row count {len(df)} does not match expected {expected_rows}")

    if list(df.columns) != ["frame", "label"]:
        raise ValueError(f"Invalid columns: expected ['frame', 'label'], got {list(df.columns)}")

    if not np.array_equal(df["frame"].values, np.arange(expected_rows)):
        raise ValueError(f"'frame' column must be contiguous integers 0..{expected_rows - 1}")

    if df["label"].isna().any():
        raise ValueError("Submission contains NaN/empty labels; write X to abstain")

    labels = set(df["label"].unique())
    valid_labels = {"H", "A", "D", "X"}
    if not labels.issubset(valid_labels):
        raise ValueError(f"Invalid labels found: {labels - valid_labels}")


def run_pipeline(
    data_dir: Path,
    output_path: Path,
    game: str = GAME,
    source: str = DEFAULT_SOURCE,
) -> pd.DataFrame:
    """Run the complete possession inference pipeline across the match.

    Args:
        data_dir: Path to match data directory.
        output_path: Path where final submission CSV should be saved.
        game: Match key (default: 'BAR-ATH').
        source: Recording source (default: 'tactical').

    Returns:
        DataFrame with predictions ['frame', 'label'].
    """
    start_time = time.perf_counter()
    print("=" * 80)
    print("La MasIA Helvetia - More Than A Hack 2027 Possession Pipeline")
    print("=" * 80)
    print(f"Match:       {game} (Source: {source})")
    print(f"Data Dir:    {data_dir}")
    print(f"Output:      {output_path}")

    # Check asset paths
    paths = asset_paths(game=game, data_dir=str(data_dir), source=source)
    meta_path = Path(paths["atd_meta"])
    raw_path = Path(paths["atd_raw"])

    if not meta_path.is_file() or not raw_path.is_file():
        raise FileNotFoundError(
            f"Required match assets missing:\n  {meta_path}\n  {raw_path}\n"
            f"Please set MTAH_DATA_DIR or place assets in {data_dir}"
        )

    n_frames = source_frames(source, game)
    pitch = GAMES[game]["pitch"]
    fps = GAMES[game]["fps"]
    half_blocks = half_video_blocks(game)

    print(f"\n[1/5] Streaming raw tracking data ({n_frames} frames, pitch {pitch[0]}x{pitch[1]} m)...")
    tracking = load_raw_tracking_data(
        raw_path=raw_path,
        meta_path=meta_path,
        n_expected_frames=n_frames,
        pitch=pitch,
    )
    raw_ball_observed = int(np.sum(~np.isnan(tracking.ball_xy[:, 0])))
    print(f"      Loaded tracking. Raw ball observed on {raw_ball_observed}/{n_frames} frames "
          f"({100.0 * raw_ball_observed / n_frames:.1f}%).")
    print(f"      Teams: H = {tracking.team_names[0]}, A = {tracking.team_names[1]}")

    print("\n[2/5] Loading Smart Tagging annotations...")
    try:
        smart_tagging_events = load_smart_tagging(game=game, data_dir=str(data_dir), fps=fps)
        print(f"      Loaded {len(smart_tagging_events)} Smart Tagging tactical intervals.")
    except Exception as e:
        print(f"      Warning: Smart Tagging could not be loaded ({e}). Proceeding with kinematics fallback.")
        smart_tagging_events = []

    print("\n[3/5] Reconstructing ball trajectory (interpolating gaps < 35 frames)...")
    ball_result = reconstruct_ball_trajectory(
        raw_ball_xy=tracking.ball_xy,
        pitch=pitch,
        max_gap_frames=35,
        max_physical_speed_mps=38.0,
        fps=fps,
    )
    reconstructed_count = int(np.sum(~ball_result.is_ball_uncertain))
    print(f"      Ball trajectory resolved on {reconstructed_count}/{n_frames} frames "
          f"({100.0 * reconstructed_count / n_frames:.1f}%).")
    print(f"      Uncertain/blackout frames: {n_frames - reconstructed_count} "
          f"({100.0 * (n_frames - reconstructed_count) / n_frames:.1f}%).")

    print("\n[4/5] Detecting Dead Ball ('D') intervals (Smart Tagging + player kinematics)...")
    dead_ball_result = detect_dead_ball(
        n_frames=n_frames,
        mean_player_speed=tracking.mean_player_speed,
        ball_xy=ball_result.ball_xy,
        is_ball_uncertain=ball_result.is_ball_uncertain,
        ball_speed=ball_result.ball_speed,
        smart_tagging_events=smart_tagging_events,
        half_blocks=half_blocks,
        pitch=pitch,
        walking_speed_threshold_mps=1.15,
        kinematics_window=35,
        min_stoppage_duration_frames=20,
        fps=fps,
    )
    dead_count = int(np.sum(dead_ball_result.is_dead_ball))
    print(f"      Dead Ball detected on {dead_count}/{n_frames} frames ({100.0 * dead_count / n_frames:.1f}%).")

    print("\n[5/5] Running temporal hysteresis possession state machine...")
    possession_result = run_possession_state_machine(
        n_frames=n_frames,
        ball_xy=ball_result.ball_xy,
        is_ball_uncertain=ball_result.is_ball_uncertain,
        is_dead_ball=dead_ball_result.is_dead_ball,
        player_xy=tracking.player_xy,
        player_speed=tracking.player_speed,
        team_of_track=tracking.team_of_track,
        smart_tagging_events=smart_tagging_events,
        half_blocks=half_blocks,
        control_radius=1.8,
        min_control_frames=22,
        scramble_abstain_frames=15,
        fps=fps,
    )

    df_submission = pd.DataFrame({
        "frame": np.arange(n_frames, dtype=np.int32),
        "label": possession_result.labels,
    })

    # Validate output format
    validate_submission_format(df_submission, expected_rows=n_frames)

    # Save output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_submission.to_csv(output_path, index=False)
    elapsed = time.perf_counter() - start_time

    # Summary metrics
    n_spells, mean_duration = count_possession_spells(possession_result.labels, half_blocks, fps=fps)

    print("\n" + "=" * 80)
    print("EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Runtime:       {elapsed:.2f} seconds")
    print(f"Output File:   {output_path} ({output_path.stat().st_size / (1024 * 1024):.2f} MB)")
    print(f"Total Rows:    {len(df_submission):,} (frames 0..{n_frames - 1})")
    print("\nClass Distribution:")
    for label in ("H", "A", "D", "X"):
        count = int(np.sum(possession_result.labels == label))
        pct = 100.0 * count / n_frames
        print(f"  {label}: {count:>8,} frames ({pct:5.1f}%)")

    print(f"\nPredicted Possession Spells: {n_spells} spells (mean duration: {mean_duration:.1f} s)")
    print("=" * 80)
    print("Pipeline completed successfully. Ready for evaluation against official ground truth.")

    return df_submission


def main():
    parser = argparse.ArgumentParser(
        description="La MasIA Helvetia - Ball Possession Identification Pipeline (More Than A Hack 2027)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to match data bundle directory (overrides MTAH_DATA_DIR / ./data)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT_DIR / "results" / "lamasia_helvetia.csv"),
        help="Path for output submission CSV (default: results/lamasia_helvetia.csv)",
    )
    parser.add_argument(
        "--game",
        type=str,
        default=GAME,
        help=f"Match identifier (default: {GAME})",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=DEFAULT_SOURCE,
        help=f"Camera source (default: {DEFAULT_SOURCE})",
    )

    args = parser.parse_args()

    data_dir = resolve_data_directory(args.data_dir)
    output_path = Path(args.output).resolve()

    run_pipeline(
        data_dir=data_dir,
        output_path=output_path,
        game=args.game,
        source=args.source,
    )


if __name__ == "__main__":
    main()
