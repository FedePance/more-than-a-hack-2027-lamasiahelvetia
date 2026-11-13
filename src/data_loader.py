"""Data loader utilities for the 'More Than A Hack 2027' possession pipeline.

Delegates frame alignment and asset resolution to official starter tooling (starter/load_atd.py)
and provides fast streaming extraction of raw Automatic Tracking Data (ATD) into structured numpy
arrays, including goalkeeper and non-outfield role filtering.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import xml.etree.ElementTree as ET

import numpy as np

# Official starter loaders and constants
from starter.load_atd import video_frame_of, atd_paths
from games import GAME, DEFAULT_SOURCE

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
    team_of_track: np.ndarray        # shape (n_slots,), int8 (0 = Home, 1 = Away)
    team_names: Tuple[str, str]      # (Home name, Away name)
    is_outfield_slot: np.ndarray     # shape (n_slots,), bool (True for outfield players)
    ball_xy: np.ndarray              # shape (n_frames, 2), float32 in metres (NaN if missing)
    ball_speed: np.ndarray           # shape (n_frames,), float32 in m/s (NaN if missing)
    player_xy: np.ndarray            # shape (n_frames, n_slots, 2), float32 in metres (NaN if missing)
    player_speed: np.ndarray         # shape (n_frames, n_slots), float32 in m/s (NaN if missing)
    mean_player_speed: np.ndarray    # shape (n_frames,), float32 in m/s (mean of active outfield players)
    p80_player_speed: np.ndarray     # shape (n_frames,), float32 in m/s (80th percentile outfield speed)


def parse_atd_metadata(meta_path: Path) -> Tuple[np.ndarray, Tuple[str, str], np.ndarray]:
    """Parse FIFA/EPTS XML metadata to extract track slot team assignments, club names, and roles.

    Follows the EPTS metadata standard:
    - Inspects MatchParameters.Score for idLocalTeam (Home) and idVisitingTeam (Away)
    - Fallback: first Team element in XML is Home (FIFA/EPTS convention)
    - Identifies Goalkeeper and non-outfield slots (position_index == 0 or name tags)

    Args:
        meta_path: Path to the ATD metadata XML file.

    Returns:
        Tuple of (team_of_track, (home_name, away_name), is_outfield_slot).
    """
    root = ET.parse(meta_path).getroot()

    # Extract team IDs and names
    team_elements = list(root.iter("Team"))
    if len(team_elements) < 2:
        raise ValueError(f"Expected at least 2 teams in metadata {meta_path}, found {len(team_elements)}")

    team_name_map = {}
    team_ids = []
    for team_elm in team_elements:
        tid = team_elm.get("id")
        tname = team_elm.findtext("Name") or f"Team_{tid}"
        team_ids.append(tid)
        team_name_map[tid] = tname

    # Determine Home team ID from Score element if available
    home_id = None
    away_id = None
    for score_elm in root.iter("Score"):
        local_id = score_elm.get("idLocalTeam")
        visiting_id = score_elm.get("idVisitingTeam")
        if local_id and visiting_id:
            home_id = local_id
            away_id = visiting_id
            break

    # Fallback to team order
    if home_id is None:
        home_id = team_ids[0]
        away_id = team_ids[1]

    home_name = team_name_map.get(home_id, "Home")
    away_name = team_name_map.get(away_id, "Away")

    player_elements = list(root.iter("Player"))
    if not player_elements:
        raise ValueError(f"No player elements found in metadata {meta_path}")

    n_slots = len(player_elements)
    team_of_track = np.full(n_slots, AWAY_CODE, dtype=np.int8)
    is_outfield_slot = np.ones(n_slots, dtype=bool)

    for slot_idx, player_elm in enumerate(player_elements):
        tid = player_elm.get("teamId")
        if tid == home_id:
            team_of_track[slot_idx] = HOME_CODE
        else:
            team_of_track[slot_idx] = AWAY_CODE

        # Role detection: check ProviderParameter for position_index == 0 (Goalkeeper)
        # or name containing Goalkeeper / GK / Referee
        name = (player_elm.findtext("Name") or "").lower()
        if "goalkeeper" in name or "gk" in name or "referee" in name or "ref" in name:
            is_outfield_slot[slot_idx] = False

        for param in player_elm.iter("ProviderParameter"):
            p_name = (param.findtext("Name") or "").lower()
            p_val = (param.findtext("Value") or "").strip()
            if p_name == "position_index" and p_val == "0":
                is_outfield_slot[slot_idx] = False
            elif p_name == "role" and ("goalkeeper" in p_val.lower() or "referee" in p_val.lower()):
                is_outfield_slot[slot_idx] = False

    return team_of_track, (home_name, away_name), is_outfield_slot


def load_raw_tracking_data(
    raw_path: Path,
    meta_path: Path,
    n_expected_frames: int,
    pitch: Tuple[float, float],
) -> RawTrackingData:
    """Stream raw ATD text file directly into structured numpy arrays.

    Delegates frame alignment to starter/load_atd.py.video_frame_of so tracking
    frames precisely align with games.py half_video_blocks and Smart Tagging.

    Args:
        raw_path: Path to the raw ATD text file.
        meta_path: Path to the ATD metadata XML file.
        n_expected_frames: Expected total frame count (e.g. 199,319).
        pitch: Pitch dimensions in metres (length, width).

    Returns:
        Populated RawTrackingData container.
    """
    team_of_track, team_names, is_outfield_slot = parse_atd_metadata(meta_path)
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
            frame_idx = video_frame_of(frame_id)
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

    # Outfield slot mask: if no slots were marked non-outfield by metadata,
    # find goalkeepers empirically (players who stay in goal area x < 12 or x > pitch_length - 12)
    outfield_indices = np.flatnonzero(is_outfield_slot)
    if len(outfield_indices) == n_slots and n_slots >= 22:
        # Detect goalkeepers by median x position near goal lines
        with np.errstate(invalid="ignore"):
            median_x = np.nanmedian(player_xy[:, :, 0], axis=0)
        potential_gk = np.flatnonzero((median_x < 14.0) | (median_x > pitch_length - 14.0))
        if len(potential_gk) >= 2:
            is_outfield_slot[potential_gk] = False
            outfield_indices = np.flatnonzero(is_outfield_slot)

    # Compute collective outfield player speed statistics (mean and 80th percentile)
    outfield_speeds = player_speed[:, outfield_indices]
    mean_player_speed = np.zeros(n_expected_frames, dtype=np.float32)
    p80_player_speed = np.zeros(n_expected_frames, dtype=np.float32)

    for i in range(n_expected_frames):
        row = outfield_speeds[i]
        valid_speeds = row[~np.isnan(row)]
        if len(valid_speeds) > 0:
            mean_player_speed[i] = float(np.mean(valid_speeds))
            p80_player_speed[i] = float(np.percentile(valid_speeds, 80))

    return RawTrackingData(
        n_frames=n_expected_frames,
        pitch=pitch,
        team_of_track=team_of_track,
        team_names=team_names,
        is_outfield_slot=is_outfield_slot,
        ball_xy=ball_xy,
        ball_speed=ball_speed,
        player_xy=player_xy,
        player_speed=player_speed,
        mean_player_speed=mean_player_speed,
        p80_player_speed=p80_player_speed,
    )
