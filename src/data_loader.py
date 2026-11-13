"""Data loader utilities for the 'More Than A Hack 2027' possession pipeline.

Provides fast, streaming extraction of raw Automatic Tracking Data (ATD) and Smart Tagging
annotations with zero heavy external dependencies.
"""

from dataclasses import dataclass
import os
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Optional, Tuple, List, Dict, Any

import numpy as np

# Domain constants
HOME_CODE = 0
AWAY_CODE = 1
FIELD_SEP = ":"
GROUP_SEP = ";"
MISSING_CHAR = "N"


@dataclass
class RawTrackingData:
    """Container for parsed match tracking arrays."""
    n_frames: int
    pitch: Tuple[float, float]
    team_of_track: np.ndarray        # shape (46,), int8 (0 = Home, 1 = Away)
    team_names: Tuple[str, str]      # (Home name, Away name)
    ball_xy: np.ndarray              # shape (n_frames, 2), float32 in metres (NaN if missing)
    ball_speed: np.ndarray           # shape (n_frames,), float32 in m/s (NaN if missing)
    player_xy: np.ndarray            # shape (n_frames, 46, 2), float32 in metres (NaN if missing)
    player_speed: np.ndarray         # shape (n_frames, 46), float32 in m/s (NaN if missing)
    mean_player_speed: np.ndarray    # shape (n_frames,), float32 in m/s (collective mean)


def parse_atd_metadata(meta_path: Path) -> Tuple[np.ndarray, Tuple[str, str]]:
    """Parse FIFA/EPTS XML metadata to extract track slot team assignments and club names.

    Args:
        meta_path: Path to the ATD metadata XML file.

    Returns:
        Tuple of (team_of_track array, (home_name, away_name)).
    """
    root = ET.parse(meta_path).getroot()
    teams = list(root.iter("Team"))
    if len(teams) < 2:
        raise ValueError(f"Expected at least 2 teams in metadata {meta_path}, found {len(teams)}")

    home_id = teams[0].get("id")
    home_name = teams[0].findtext("Name") or "Home"
    away_name = teams[1].findtext("Name") or "Away"

    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError(f"No player slots defined in {meta_path}")

    team_of_track = np.array(
        [HOME_CODE if slot_team == home_id else AWAY_CODE for slot_team in slots],
        dtype=np.int8,
    )
    return team_of_track, (home_name, away_name)


def load_raw_tracking_data(
    raw_path: Path,
    meta_path: Path,
    n_expected_frames: int,
    pitch: Tuple[float, float],
) -> RawTrackingData:
    """Stream raw ATD text file directly into structured numpy arrays.

    Coordinates are scaled to metres using the provided pitch dimensions (length, width).

    Args:
        raw_path: Path to the raw ATD text file.
        meta_path: Path to the ATD metadata XML file.
        n_expected_frames: Expected total frame count (e.g. 199,319).
        pitch: Pitch dimensions in metres (length, width).

    Returns:
        Populated RawTrackingData container.
    """
    team_of_track, team_names = parse_atd_metadata(meta_path)
    n_slots = len(team_of_track)
    pitch_length, pitch_width = pitch

    # Pre-allocate contiguous arrays
    ball_xy = np.full((n_expected_frames, 2), np.nan, dtype=np.float32)
    ball_speed = np.full(n_expected_frames, np.nan, dtype=np.float32)
    player_xy = np.full((n_expected_frames, n_slots, 2), np.nan, dtype=np.float32)
    player_speed = np.full((n_expected_frames, n_slots), np.nan, dtype=np.float32)

    with open(raw_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split(FIELD_SEP)
            if len(parts) != 3:
                continue

            frame_id = int(parts[0])
            frame_idx = frame_id - 1  # 1-indexed in file to 0-indexed in video frames
            if frame_idx < 0 or frame_idx >= n_expected_frames:
                continue

            # Parse ball: "x,y,speed"
            ball_str = parts[2]
            if ball_str and ball_str[0] != MISSING_CHAR:
                c1 = ball_str.find(",")
                if c1 != -1:
                    c2 = ball_str.find(",", c1 + 1)
                    if c2 != -1:
                        ball_xy[frame_idx, 0] = float(ball_str[:c1]) * pitch_length
                        ball_xy[frame_idx, 1] = float(ball_str[c1 + 1 : c2]) * pitch_width
                        ball_speed[frame_idx] = float(ball_str[c2 + 1 :])

            # Parse players: "x,y,speed;x,y,speed;..."
            player_groups = parts[1].split(GROUP_SEP)
            for slot_idx, group in enumerate(player_groups):
                if slot_idx >= n_slots:
                    break
                if group and group[0] != MISSING_CHAR:
                    c1 = group.find(",")
                    if c1 != -1:
                        c2 = group.find(",", c1 + 1)
                        if c2 != -1:
                            player_xy[frame_idx, slot_idx, 0] = float(group[:c1]) * pitch_length
                            player_xy[frame_idx, slot_idx, 1] = float(group[c1 + 1 : c2]) * pitch_width
                            player_speed[frame_idx, slot_idx] = float(group[c2 + 1 :])

    # Compute collective mean speed per frame across active outfield players
    with np.errstate(invalid="ignore"):
        mean_player_speed = np.nanmean(player_speed, axis=1).astype(np.float32)
    # Replace all-NaN rows with 0.0
    mean_player_speed = np.nan_to_num(mean_player_speed, nan=0.0)

    return RawTrackingData(
        n_frames=n_expected_frames,
        pitch=pitch,
        team_of_track=team_of_track,
        team_names=team_names,
        ball_xy=ball_xy,
        ball_speed=ball_speed,
        player_xy=player_xy,
        player_speed=player_speed,
        mean_player_speed=mean_player_speed,
    )
