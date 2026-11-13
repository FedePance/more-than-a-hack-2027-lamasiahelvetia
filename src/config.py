"""Centralized configuration for the 'La MasIA Helvetia' possession identification pipeline.

All thresholds are grounded strictly in football physical principles and video frame rate (25 fps),
ensuring robust performance across match data without fitting on public ground truth.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class PipelineConfig:
    """Typed configuration container for all physical and temporal parameters."""

    # Video & pitch geometry
    fps: float = 25.0
    pitch_length: float = 105.3
    pitch_width: float = 68.0

    # Ball trajectory reconstruction & kinematics
    max_ball_gap_frames: int = 35           # 1.4 s at 25 fps: max occlusion duration to interpolate
    max_ball_speed_mps: float = 38.0        # Physical limit of elite ball speed (~137 km/h)
    spike_speed_threshold_mps: float = 40.0  # Teleportation spike detection threshold
    savgol_window: int = 7                  # Light Savitzky-Golay smoothing window (odd integer)
    savgol_polyorder: int = 2               # Polynomial order for Savitzky-Golay filter

    # Dead ball detection
    walking_speed_threshold_mps: float = 1.15   # Collective walking speed (m/s)
    jogging_speed_threshold_mps: float = 1.40   # 80th-percentile outfield player speed threshold (m/s)
    kinematic_stoppage_frames: int = 50         # 2.0 s sustained low speed required for kinematic D
    dilation_window_frames: int = 125           # 5.0 s dilation around Smart Tagging stoppages
    min_stoppage_duration_frames: int = 20      # 0.8 s minimum duration for valid stoppage
    stationary_ball_speed_mps: float = 0.35     # Ball considered stationary below this speed
    abandoned_ball_distance_m: float = 2.5      # Distance from players to flag stationary ball

    # Possession engine & hysteresis state machine
    control_radius: float = 1.8                 # Proximity threshold (m) for player ball control
    min_control_frames: int = 22                # Sustained control duration (~0.88 s) to change possession
    scramble_abstain_frames: int = 15           # Contested duel duration before emitting 'X'
    impulse_speed_threshold_mps: float = 5.0    # Ball speed indicating clearance/interception impulse
    impulse_proximity_m: float = 1.6            # Proximity required for one-touch turnover attribution
    restart_search_frames: int = 50             # Forward window (2.0 s) to locate restart taker

    @property
    def pitch(self) -> Tuple[float, float]:
        """Return pitch dimensions as (length, width) tuple."""
        return (self.pitch_length, self.pitch_width)
