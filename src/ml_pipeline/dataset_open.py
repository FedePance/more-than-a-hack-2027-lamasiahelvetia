"""Open data ingestion (Kloppy / EPTS / CSV), Ground Truth labeling, and PyTorch Dataset.

Loads Metrica Sample Games 1-3, precomputes canonical 14-dimensional base feature
representations once at load time, caches parsed games to .npz for fast startup (< 0.2 s),
and implements on-the-fly domain augmentations directly on precomputed slices:
1. 41% Ball Occlusion Dropout + Tracking Jitter (sigma = 0.25 m).
2. Random Home <-> Away Swap (p = 0.5) with label transposition.
3. Pitch Coordinate Reflection (p = 0.5 each axis).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.ml_pipeline.features import (
    extract_frame_features,
)


@dataclass
class OpenGameData:
    """Container holding unified tracking, kinematics, events, and labels for one match."""

    game_id: str
    n_frames: int
    fps: float
    pitch: tuple[float, float]
    ball_xy: np.ndarray  # (n_frames, 2), in metres
    ball_speed: np.ndarray  # (n_frames,), in m/s
    is_ball_visible: np.ndarray  # (n_frames,), bool
    home_player_xy: np.ndarray  # (n_frames, n_home, 2), in metres
    home_player_speed: np.ndarray  # (n_frames, n_home), in m/s
    away_player_xy: np.ndarray  # (n_frames, n_away, 2), in metres
    away_player_speed: np.ndarray  # (n_frames, n_away), in m/s
    outfield_mean_speed: np.ndarray  # (n_frames,), in m/s
    outfield_p80_speed: np.ndarray  # (n_frames,), in m/s
    home_centroid_x_norm: np.ndarray  # (n_frames,), in [-1.0, 1.0]
    away_centroid_x_norm: np.ndarray  # (n_frames,), in [-1.0, 1.0]
    is_stoppage_tag: np.ndarray  # (n_frames,), float32
    gt_labels: np.ndarray  # (n_frames,), int8 (0=H, 1=A, 2=D)
    features: np.ndarray  # (n_frames, 14), float32 precomputed base feature matrix


def locate_open_data_dir(data_dir: str | Path | None = None) -> Path:
    """Find the root directory containing Metrica Sample Games."""
    candidates = []
    if data_dir is not None:
        candidates.append(Path(data_dir))
    candidates.extend([
        Path("external_open/data"),
        Path("external_open/sample-data-master/data"),
        Path("../external_open/data"),
        Path("/workspaces/barca-hackathon/external_open/data"),
    ])
    for p in candidates:
        if p.is_dir() and (p / "Sample_Game_1").is_dir():
            return p
    raise FileNotFoundError(
        f"Could not locate open data directory. Checked candidates: {[str(c) for c in candidates]}"
    )


def generate_ball_dropout_mask(
    n_frames: int,
    target_ratio: float = 0.41,
    min_span: int = 5,
    max_span: int = 75,
    seed: int = 42,
) -> np.ndarray:
    """Generate a deterministic boolean mask simulating the realistic ~41% missing ball rate.

    Args:
        n_frames: Total number of frames.
        target_ratio: Target fraction of occluded frames (default: 0.41).
        min_span: Minimum contiguous occlusion duration in frames (default: 5 frames / 0.2 s).
        max_span: Maximum contiguous occlusion duration in frames (default: 75 frames / 3.0 s).
        seed: Random seed for deterministic reproducibility (default: 42).

    Returns:
        np.ndarray of shape (n_frames,), dtype bool (True where ball is masked/occluded).
    """
    rng = np.random.RandomState(seed)
    mask = np.zeros(n_frames, dtype=bool)
    target_dropped = round(n_frames * target_ratio)
    current_dropped = 0
    max_attempts = n_frames * 3
    attempts = 0

    while current_dropped < target_dropped and attempts < max_attempts:
        attempts += 1
        span_len = rng.randint(min_span, max_span + 1)
        start_idx = rng.randint(0, max(1, n_frames - span_len))
        end_idx = min(n_frames, start_idx + span_len)

        new_dropped = int((~mask[start_idx:end_idx]).sum())
        mask[start_idx:end_idx] = True
        current_dropped += new_dropped

    return mask


def _compute_central_diff_speed(
    coords: np.ndarray, dt: float = 0.04, max_speed: float = 40.0
) -> np.ndarray:
    """Compute central difference velocity from (n_frames, ..., 2) coordinates."""
    has_players = coords.ndim == 3
    if not has_players:
        coords = coords[:, None, :]

    n_frames, n_entities, _ = coords.shape
    speeds = np.full((n_frames, n_entities), np.nan, dtype=np.float32)

    if n_frames < 2:
        return speeds if has_players else speeds[:, 0]

    # Interior frames: central difference (2 * dt)
    dx = coords[2:, :, 0] - coords[:-2, :, 0]
    dy = coords[2:, :, 1] - coords[:-2, :, 1]
    speeds[1:-1] = np.sqrt(dx * dx + dy * dy) / (2.0 * dt)

    # Boundary frames: forward/backward difference
    dx0 = coords[1, :, 0] - coords[0, :, 0]
    dy0 = coords[1, :, 1] - coords[0, :, 1]
    speeds[0] = np.sqrt(dx0 * dx0 + dy0 * dy0) / dt

    dx_end = coords[-1, :, 0] - coords[-2, :, 0]
    dy_end = coords[-1, :, 1] - coords[-2, :, 1]
    speeds[-1] = np.sqrt(dx_end * dx_end + dy_end * dy_end) / dt

    speeds = np.clip(speeds, 0.0, max_speed)
    return speeds if has_players else speeds[:, 0]


def _build_gt_labels_and_stoppages(
    events_list: list[dict],
    n_frames: int,
    period_bounds: dict[int, tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Derive exact frame-level Ground Truth labels and stoppage masks from events.

    Labels:
        0 = Home possession ('H')
        1 = Away possession ('A')
        2 = Dead Ball / Out of Play ('D')

    Args:
        events_list: List of parsed event dictionaries sorted by frame.
        n_frames: Total match frames.
        period_bounds: Mapping from period ID (1, 2) to (first_frame, last_frame).

    Returns:
        Tuple of (gt_labels, is_stoppage_tag) arrays of shape (n_frames,).
    """
    gt_labels = np.full(n_frames, 2, dtype=np.int8)  # default Dead Ball (D)
    is_stoppage = np.zeros(n_frames, dtype=np.float32)

    # Sort events chronologically
    sorted_events = sorted(events_list, key=lambda e: (e.get("period", 1), e.get("start_frame", 0)))

    for i, ev in enumerate(sorted_events):
        p = ev.get("period", 1)
        sf = ev.get("start_frame", 0)
        ef = ev.get("end_frame", sf)
        etype = (ev.get("type") or "").upper().strip()
        subtype = (ev.get("subtype") or "").upper().strip()
        team = (ev.get("team") or "").upper().strip()

        sf = max(0, min(n_frames - 1, sf))
        ef = max(sf, min(n_frames - 1, ef))

        if etype in ("SET PIECE", "BALL OUT"):
            next_start = ef
            if i + 1 < len(sorted_events) and sorted_events[i + 1].get("period", 1) == p:
                next_start = min(n_frames - 1, sorted_events[i + 1].get("start_frame", ef))

            stoppage_end = max(ef, next_start)
            gt_labels[sf:stoppage_end] = 2  # D
            is_stoppage[sf:stoppage_end] = 1.0

        elif etype in ("PASS", "RECOVERY", "SHOT", "CHALLENGE", "CARRY", "BALL LOST"):
            team_code = 0 if ("HOME" in team or team == "TEAM A" or "FIFATMA" in team) else 1

            next_sf = ef
            if i + 1 < len(sorted_events) and sorted_events[i + 1].get("period", 1) == p:
                next_sf = min(n_frames - 1, sorted_events[i + 1].get("start_frame", ef))

            spell_end = max(ef, next_sf)
            gt_labels[sf:spell_end] = team_code

            if "GOAL" in subtype:
                gt_labels[ef:spell_end] = 2  # D
                is_stoppage[ef:spell_end] = 1.0

    # Ensure pre-match, halftime, and post-match periods are strictly Dead Ball (D)
    for period, (p_start, p_end) in period_bounds.items():
        if period == 1:
            gt_labels[0:p_start] = 2
            is_stoppage[0:p_start] = 1.0
        elif period == 2:
            prev_end = period_bounds[1][1]
            gt_labels[prev_end:p_start] = 2
            is_stoppage[prev_end:p_start] = 1.0
            gt_labels[p_end:n_frames] = 2
            is_stoppage[p_end:n_frames] = 1.0

    return gt_labels, is_stoppage


def load_open_game_data(
    game_id_or_name: str | int,
    data_dir: str | Path | None = None,
    pitch: tuple[float, float] = (105.3, 68.0),
    fps: float = 25.0,
    use_cache: bool = True,
) -> OpenGameData:
    """Load Sample Game 1, 2, or 3 from open data repository into unified container.

    Checks external_open/.cache/{game_str}.npz first for instant (< 0.2 s) loading.

    Args:
        game_id_or_name: 1, 2, 3 or "Sample_Game_1", "Sample_Game_2", "Sample_Game_3".
        data_dir: Optional open data directory path.
        pitch: Pitch dimensions in metres (length, width). Default: (105.3, 68.0).
        fps: Sampling rate (default: 25.0).
        use_cache: Whether to read/write cached .npz files.

    Returns:
        Populated OpenGameData container.
    """
    root_dir = locate_open_data_dir(data_dir)
    pitch_len, pitch_wid = pitch
    dt = 1.0 / fps

    if str(game_id_or_name) in ("1", "Sample_Game_1"):
        game_folder = root_dir / "Sample_Game_1"
        game_str = "Sample_Game_1"
        is_epts = False
    elif str(game_id_or_name) in ("2", "Sample_Game_2"):
        game_folder = root_dir / "Sample_Game_2"
        game_str = "Sample_Game_2"
        is_epts = False
    elif str(game_id_or_name) in ("3", "Sample_Game_3"):
        game_folder = root_dir / "Sample_Game_3"
        game_str = "Sample_Game_3"
        is_epts = True
    else:
        raise ValueError(f"Unknown game identifier: {game_id_or_name}")

    # Check cache directory
    cache_dir = root_dir.parent / ".cache"
    cache_file = cache_dir / f"{game_str}.npz"

    if use_cache and cache_file.is_file():
        try:
            cached = np.load(cache_file, allow_pickle=False)
            return OpenGameData(
                game_id=game_str,
                n_frames=int(cached["n_frames"]),
                fps=float(cached["fps"]),
                pitch=tuple(float(x) for x in cached["pitch"]),
                ball_xy=cached["ball_xy"],
                ball_speed=cached["ball_speed"],
                is_ball_visible=cached["is_ball_visible"],
                home_player_xy=cached["home_player_xy"],
                home_player_speed=cached["home_player_speed"],
                away_player_xy=cached["away_player_xy"],
                away_player_speed=cached["away_player_speed"],
                outfield_mean_speed=cached["outfield_mean_speed"],
                outfield_p80_speed=cached["outfield_p80_speed"],
                home_centroid_x_norm=cached["home_centroid_x_norm"],
                away_centroid_x_norm=cached["away_centroid_x_norm"],
                is_stoppage_tag=cached["is_stoppage_tag"],
                gt_labels=cached["gt_labels"],
                features=cached["features"],
            )
        except (OSError, KeyError, ValueError):
            pass  # Fall through to raw parsing if cache corrupted

    if not is_epts:
        # Load Metrica CSV Tracking Data (Games 1 and 2)
        home_csv = game_folder / f"{game_str}_RawTrackingData_Home_Team.csv"
        away_csv = game_folder / f"{game_str}_RawTrackingData_Away_Team.csv"
        events_csv = game_folder / f"{game_str}_RawEventsData.csv"

        df_home = pd.read_csv(home_csv, skiprows=2)
        df_away = pd.read_csv(away_csv, skiprows=2)

        n_frames = len(df_home)
        periods = df_home["Period"].values

        # Ball coordinates: in Metrica CSV, the last two columns are Ball X and Ball Y
        ball_x = df_home.iloc[:, -2].values.astype(np.float32) * pitch_len
        ball_y = df_home.iloc[:, -1].values.astype(np.float32) * pitch_wid
        ball_xy = np.column_stack([ball_x, ball_y])
        is_ball_visible = ~np.isnan(ball_xy[:, 0])

        # Extract Home outfield player coordinates (columns 3 to -2 in pairs, excluding Player 11 GK)
        h_outfield_cols = []
        for c in range(3, df_home.shape[1] - 2, 2):
            col_name = str(df_home.columns[c]).strip().lower().replace(" ", "")
            if "11" not in col_name and "gk" not in col_name:
                h_outfield_cols.append(c)

        home_xy = np.zeros((n_frames, len(h_outfield_cols), 2), dtype=np.float32)
        for i, c in enumerate(h_outfield_cols):
            home_xy[:, i, 0] = df_home.iloc[:, c].values.astype(np.float32) * pitch_len
            home_xy[:, i, 1] = df_home.iloc[:, c + 1].values.astype(np.float32) * pitch_wid

        # Extract Away outfield player coordinates (columns 3 to -2 in pairs, excluding Player 25 GK)
        a_outfield_cols = []
        for c in range(3, df_away.shape[1] - 2, 2):
            col_name = str(df_away.columns[c]).strip().lower().replace(" ", "")
            if "25" not in col_name and "gk" not in col_name:
                a_outfield_cols.append(c)

        away_xy = np.zeros((n_frames, len(a_outfield_cols), 2), dtype=np.float32)
        for i, c in enumerate(a_outfield_cols):
            away_xy[:, i, 0] = df_away.iloc[:, c].values.astype(np.float32) * pitch_len
            away_xy[:, i, 1] = df_away.iloc[:, c + 1].values.astype(np.float32) * pitch_wid

        # Events
        df_events = pd.read_csv(events_csv)
        events_list = []
        for _, row in df_events.iterrows():
            events_list.append({
                "period": int(row["Period"]),
                "start_frame": int(row["Start Frame"]),
                "end_frame": int(row["End Frame"]),
                "type": str(row["Type"]),
                "subtype": str(row.get("Subtype", "")),
                "team": str(row.get("Team", "")),
            })

        p1_idx = np.flatnonzero(periods == 1)
        p2_idx = np.flatnonzero(periods == 2)
        period_bounds = {
            1: (int(p1_idx[0]), int(p1_idx[-1])),
            2: (int(p2_idx[0]), int(p2_idx[-1])),
        }

    else:
        # Load Metrica EPTS Data (Game 3)
        tracking_txt = game_folder / f"{game_str}_tracking.txt"
        events_json = game_folder / f"{game_str}_events.json"

        frames_data = []
        with open(tracking_txt, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(":")
                if len(parts) >= 3:
                    f_id = int(parts[0])
                    p_str = parts[1].split(";")
                    b_str = parts[2].split(",")
                    frames_data.append((f_id, p_str, b_str))

        n_frames = len(frames_data)
        ball_xy = np.full((n_frames, 2), np.nan, dtype=np.float32)

        # In Game 3: 22 tracking slots (slots 0..10 Team A, slots 11..21 Team B)
        # GK: slot 0 (Player 11) is Team A GK; slot 11 (Player 28) is Team B GK
        n_h_outfield = 10
        n_a_outfield = 10
        home_xy = np.full((n_frames, n_h_outfield, 2), np.nan, dtype=np.float32)
        away_xy = np.full((n_frames, n_a_outfield, 2), np.nan, dtype=np.float32)

        for idx, (f_id, p_str, b_str) in enumerate(frames_data):
            if len(b_str) >= 2 and b_str[0] not in ("NaN", "-1", "N"):
                try:
                    ball_xy[idx, 0] = float(b_str[0]) * pitch_len
                    ball_xy[idx, 1] = float(b_str[1]) * pitch_wid
                except ValueError:
                    pass

            for s_idx, p_data in enumerate(p_str):
                c = p_data.split(",")
                if len(c) >= 2 and c[0] not in ("NaN", "-1", "N"):
                    try:
                        px = float(c[0]) * pitch_len
                        py = float(c[1]) * pitch_wid
                        if 1 <= s_idx <= 10:  # Team A outfield
                            home_xy[idx, s_idx - 1, 0] = px
                            home_xy[idx, s_idx - 1, 1] = py
                        elif 12 <= s_idx <= 21:  # Team B outfield
                            away_xy[idx, s_idx - 12, 0] = px
                            away_xy[idx, s_idx - 12, 1] = py
                    except ValueError:
                        pass

        is_ball_visible = ~np.isnan(ball_xy[:, 0])

        with open(events_json, "r", encoding="utf-8") as f:
            raw_ev_data = json.load(f)["data"]

        events_list = []
        for ev in raw_ev_data:
            events_list.append({
                "period": ev.get("period", 1),
                "start_frame": ev.get("start", {}).get("frame", 0),
                "end_frame": ev.get("end", {}).get("frame", 0),
                "type": ev.get("type", {}).get("name", ""),
                "subtype": ev.get("subtypes", {}).get("name") if isinstance(ev.get("subtypes"), dict) else "",
                "team": ev.get("team", {}).get("name", ""),
            })

        period_bounds = {1: (361, 69594), 2: (70090, 143553)}

    # Compute speeds via central differences
    ball_speed = _compute_central_diff_speed(ball_xy, dt=dt, max_speed=38.0)
    home_player_speed = _compute_central_diff_speed(home_xy, dt=dt, max_speed=12.0)
    away_player_speed = _compute_central_diff_speed(away_xy, dt=dt, max_speed=12.0)

    # Compute collective outfield speeds
    comb_speeds = np.concatenate([home_player_speed, away_player_speed], axis=1)
    with np.errstate(invalid="ignore"):
        outfield_mean_speed = np.nan_to_num(np.nanmean(comb_speeds, axis=1), nan=0.0).astype(np.float32)
        outfield_p80_speed = np.nan_to_num(np.nanpercentile(comb_speeds, 80, axis=1), nan=0.0).astype(np.float32)

    # Compute team longitudinal centroids
    half_len = pitch_len / 2.0
    with np.errstate(invalid="ignore"):
        h_cx = (np.nanmean(home_xy[:, :, 0], axis=1) - half_len) / half_len
        a_cx = (np.nanmean(away_xy[:, :, 0], axis=1) - half_len) / half_len
    home_centroid_x_norm = np.clip(np.nan_to_num(h_cx, nan=0.0), -1.0, 1.0).astype(np.float32)
    away_centroid_x_norm = np.clip(np.nan_to_num(a_cx, nan=0.0), -1.0, 1.0).astype(np.float32)

    # Derive GT labels and stoppage tag mask
    gt_labels, is_stoppage_tag = _build_gt_labels_and_stoppages(events_list, n_frames, period_bounds)

    # Precompute canonical 14-dimensional base feature representations once
    features = extract_frame_features(
        ball_xy=ball_xy,
        ball_speed=ball_speed,
        is_ball_visible=is_ball_visible,
        home_player_xy=home_xy,
        home_player_speed=home_player_speed,
        away_player_xy=away_xy,
        away_player_speed=away_player_speed,
        is_stoppage_tag=is_stoppage_tag,
        pitch=pitch,
        outfield_mean_speed=outfield_mean_speed,
        outfield_p80_speed=outfield_p80_speed,
        home_centroid_x_norm=home_centroid_x_norm,
        away_centroid_x_norm=away_centroid_x_norm,
    )

    # Cache precomputed arrays to .npz for fast future loads
    if use_cache:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_file,
                n_frames=n_frames,
                fps=fps,
                pitch=np.array(pitch, dtype=np.float32),
                ball_xy=ball_xy,
                ball_speed=ball_speed,
                is_ball_visible=is_ball_visible,
                home_player_xy=home_xy,
                home_player_speed=home_player_speed,
                away_player_xy=away_xy,
                away_player_speed=away_player_speed,
                outfield_mean_speed=outfield_mean_speed,
                outfield_p80_speed=outfield_p80_speed,
                home_centroid_x_norm=home_centroid_x_norm,
                away_centroid_x_norm=away_centroid_x_norm,
                is_stoppage_tag=is_stoppage_tag,
                gt_labels=gt_labels,
                features=features,
            )
        except (OSError, ValueError):
            pass

    return OpenGameData(
        game_id=game_str,
        n_frames=n_frames,
        fps=fps,
        pitch=pitch,
        ball_xy=ball_xy,
        ball_speed=ball_speed,
        is_ball_visible=is_ball_visible,
        home_player_xy=home_xy,
        home_player_speed=home_player_speed,
        away_player_xy=away_xy,
        away_player_speed=away_player_speed,
        outfield_mean_speed=outfield_mean_speed,
        outfield_p80_speed=outfield_p80_speed,
        home_centroid_x_norm=home_centroid_x_norm,
        away_centroid_x_norm=away_centroid_x_norm,
        is_stoppage_tag=is_stoppage_tag,
        gt_labels=gt_labels,
        features=features,
    )


class OpenGameDataset(Dataset):
    """PyTorch Dataset slicing sequences with fast on-the-fly domain augmentations."""

    def __init__(
        self,
        games: list[OpenGameData],
        seq_len: int = 250,
        stride: int = 125,
        augment: bool = True,
        dropout_ratio: float = 0.41,
        coord_jitter_sigma: float = 0.25,
    ):
        """Initialize dataset across one or more open games.

        Args:
            games: List of OpenGameData containers with precomputed features.
            seq_len: Sequence length in frames (default: 250 frames / 10.0 s).
            stride: Stride between slices (default: 125 frames for 50% overlap).
            augment: Whether to apply domain augmentations on the fly.
            dropout_ratio: Target ball occlusion dropout ratio (default: 0.41).
            coord_jitter_sigma: Coordinate noise standard deviation in metres (default: 0.25 m).
        """
        self.games = games
        self.seq_len = seq_len
        self.stride = stride
        self.augment = augment
        self.dropout_ratio = dropout_ratio
        self.jitter_sigma = coord_jitter_sigma

        # Precompute slice indices: list of (game_idx, start_frame)
        self.slices: list[tuple[int, int]] = []
        for g_idx, game in enumerate(self.games):
            max_start = game.n_frames - seq_len
            for start in range(0, max_start + 1, stride):
                self.slices.append((g_idx, start))

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        g_idx, start = self.slices[idx]
        game = self.games[g_idx]
        end = start + self.seq_len

        # Fast direct slice from precomputed base feature matrix and labels
        feat = game.features[start:end].copy()
        labels = game.gt_labels[start:end].copy()

        if self.augment:
            # 1. 41% Ball Occlusion Dropout + Tracking Jitter
            # Synthetically mask contiguous ball spans
            drop_mask = np.zeros(self.seq_len, dtype=bool)
            target_drop = int(self.seq_len * self.dropout_ratio)
            n_dropped = 0
            attempts = 0
            while n_dropped < target_drop and attempts < 10:
                attempts += 1
                span_len = np.random.randint(5, 76)
                s_idx = np.random.randint(0, max(1, self.seq_len - span_len))
                e_idx = min(self.seq_len, s_idx + span_len)
                n_dropped += int((~drop_mask[s_idx:e_idx]).sum())
                drop_mask[s_idx:e_idx] = True

            # Where masked or originally missing: reset ball features to sentinel/zeros
            masked = drop_mask | (feat[:, 3] < 0.5)
            feat[masked, 0] = 50.0  # min_dist_home
            feat[masked, 1] = 50.0  # min_dist_away
            feat[masked, 2] = 0.0   # dist_diff
            feat[masked, 3] = 0.0   # is_ball_visible
            feat[masked, 4] = 0.0   # ball_speed
            feat[masked, 5] = 0.0   # ball_x_norm
            feat[masked, 6] = 0.0   # ball_y_norm
            feat[masked, 7] = 0.0   # closest_home_speed
            feat[masked, 8] = 0.0   # closest_away_speed

            # On frames where ball remains visible, add jitter
            vis = ~masked
            n_vis = int(vis.sum())
            if n_vis > 0:
                jitter_h = np.random.normal(0.0, self.jitter_sigma, size=n_vis).astype(np.float32)
                jitter_a = np.random.normal(0.0, self.jitter_sigma, size=n_vis).astype(np.float32)
                feat[vis, 0] = np.maximum(0.1, feat[vis, 0] + jitter_h)
                feat[vis, 1] = np.maximum(0.1, feat[vis, 1] + jitter_a)
                feat[vis, 2] = feat[vis, 1] - feat[vis, 0]

                # Ball normalized coordinates jitter
                jx_norm = np.random.normal(0.0, self.jitter_sigma / 52.65, size=n_vis).astype(np.float32)
                jy_norm = np.random.normal(0.0, self.jitter_sigma / 34.0, size=n_vis).astype(np.float32)
                feat[vis, 5] = np.clip(feat[vis, 5] + jx_norm, -1.0, 1.0)
                feat[vis, 6] = np.clip(feat[vis, 6] + jy_norm, -1.0, 1.0)

                # Ball speed light jitter
                spd_j = np.random.normal(0.0, 0.2, size=n_vis).astype(np.float32)
                feat[vis, 4] = np.maximum(0.0, feat[vis, 4] + spd_j)

            # 2. Random Home <-> Away Swap (p = 0.5)
            if np.random.rand() < 0.5:
                # Swap min_dist_home (0) and min_dist_away (1)
                tmp0 = feat[:, 0].copy()
                feat[:, 0] = feat[:, 1]
                feat[:, 1] = tmp0
                # Negate dist_diff (2)
                feat[:, 2] = -feat[:, 2]
                # Swap closest player speeds (7 and 8)
                tmp7 = feat[:, 7].copy()
                feat[:, 7] = feat[:, 8]
                feat[:, 8] = tmp7
                # Swap team longitudinal centroids (11 and 12)
                tmp11 = feat[:, 11].copy()
                feat[:, 11] = feat[:, 12]
                feat[:, 12] = tmp11

                # Transpose labels: 0 <-> 1 (leaving 2 unchanged)
                swapped = labels.copy()
                swapped[labels == 0] = 1
                swapped[labels == 1] = 0
                labels = swapped

            # 3. Pitch Coordinate Reflection (p = 0.5 each axis)
            if np.random.rand() < 0.5:
                # Mirror X: ball_x_norm (5), home_centroid_x_norm (11), away_centroid_x_norm (12)
                feat[:, 5] = -feat[:, 5]
                feat[:, 11] = -feat[:, 11]
                feat[:, 12] = -feat[:, 12]

            if np.random.rand() < 0.5:
                # Mirror Y: ball_y_norm (6)
                feat[:, 6] = -feat[:, 6]

        return torch.from_numpy(feat).float(), torch.from_numpy(labels).long()
