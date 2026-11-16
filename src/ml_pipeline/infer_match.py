"""Inference CLI for competition match (BAR-ATH).

Loads match assets via games.py, src/data_loader.py, src/ball_tracking.py,
and starter/load_smart_tagging.py, extracts canonical 14-dimensional features,
runs model inference (single checkpoint, heterogeneous ensemble, or LOGO 3-fold ensemble),
optionally fuses physical dead-ball detections with neural emission distributions,
decodes each half independently via Viterbi decoding, validates format,
and writes submission to results/lamasia_helvetia_*.csv.

Halves are processed in complete isolation: sliding windows, temporal feature expansion and
Viterbi state never cross the halftime break, and every frame outside the two halves
(pre-match, halftime, post-match) is written as Dead Ball ('D').
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

# Add project root and starter/evaluation directories to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))
sys.path.insert(0, str(PROJECT_ROOT / "starter"))

import possession_metrics as pm
from validate_submission import describe

from games import (
    DATA_DIR,
    GAME,
    GAMES,
    half_video_blocks,
    require,
)
from src.ball_tracking import reconstruct_ball_trajectory
from src.config import PipelineConfig
from src.data_loader import load_raw_tracking_data
from src.dead_ball import build_stoppage_tag_mask, detect_dead_ball
from src.ml_pipeline.features import (
    BALL_VISIBLE_INDEX,
    extract_frame_features_from_tracking,
)
from src.ml_pipeline.inference import (
    N_CLASSES,
    load_model_from_checkpoint,
    predict_sliding_window_probabilities,
    resolve_device,
    seed_everything,
    validate_window_args,
)
from src.ml_pipeline.models import ARCH_ALIASES, PossessionLGBM
from src.ml_pipeline.viterbi import LABEL_TO_CODE, codes_to_labels, viterbi_decode
from starter.load_smart_tagging import load as load_smart_tagging_events

DEAD_BALL_CODE = LABEL_TO_CODE["D"]
# Probability floor before taking logs: keeps every emission finite (log(1e-7) ~ -16.1)
MIN_EMISSION_PROB = 1e-7
INFERENCE_SEED = 42
N_LOGO_FOLDS = 3


def predict_half_probabilities(
    models: list[nn.Module | PossessionLGBM],
    weights: list[float] | np.ndarray,
    features: np.ndarray,
    device: torch.device,
    seq_len: int = 250,
    stride: int = 125,
    batch_size: int = 64,
) -> np.ndarray:
    """Predict probabilities across a half, computing weighted average across models.

    Returns:
        Weighted probabilities of shape (n_frames, 3), dtype float32.
    """
    n_frames = len(features)
    weighted_probs = np.zeros((n_frames, N_CLASSES), dtype=np.float32)

    for model, w in zip(models, weights):
        if isinstance(model, PossessionLGBM):
            m_probs = model.predict_proba(features)
        else:
            m_probs = predict_sliding_window_probabilities(
                model=model,
                features=features,
                device=device,
                seq_len=seq_len,
                stride=stride,
                batch_size=batch_size,
            )
        weighted_probs += float(w) * m_probs

    return weighted_probs


def fuse_dead_ball_evidence(
    probs: np.ndarray,
    is_dead: np.ndarray | None = None,
    dead_ball_boost: float = 0.0,
) -> np.ndarray:
    """Convert ensemble probabilities to log-scores, optionally fused with physical D evidence.

    Probabilities are floored at MIN_EMISSION_PROB before the log, so the result is always
    finite. When a physical dead-ball mask is given, the Dead Ball log-score is raised by
    dead_ball_boost on flagged frames and lowered by half of it elsewhere. The result is an
    unnormalized log-score matrix: decode it with emission_type='logits'.

    Args:
        probs: Ensemble probabilities of shape (n_frames, 3) for states (H, A, D).
        is_dead: Optional boolean mask of shape (n_frames,) from the kinematic detector.
        dead_ball_boost: Non-negative log-score boost for the Dead Ball state.

    Returns:
        Log-score matrix of shape (n_frames, 3), dtype float32.
    """
    log_scores = np.log(np.clip(probs, MIN_EMISSION_PROB, 1.0)).astype(np.float32)
    if is_dead is not None:
        is_dead = np.asarray(is_dead, dtype=bool)
        log_scores[is_dead, DEAD_BALL_CODE] += dead_ball_boost
        log_scores[~is_dead, DEAD_BALL_CODE] -= 0.5 * dead_ball_boost
    return log_scores


def resolve_checkpoints(args: argparse.Namespace) -> tuple[list[Path], list[float]]:
    """Resolve the checkpoint list and normalized ensemble weights from CLI arguments.

    Returns:
        Tuple (checkpoint_paths, normalized_weights); the weights sum to 1.

    Raises:
        FileNotFoundError: If any checkpoint file is missing.
    """
    weights = None
    if args.checkpoints:
        checkpoint_paths = [Path(p) for p in args.checkpoints]
        if args.weights:
            weights = [float(w) for w in args.weights]
    elif args.ensemble:
        models_dir = Path(args.models_dir)
        if ARCH_ALIASES[args.arch] == "lgbm":
            names = [f"lgbm_fold_{k}.joblib" for k in range(1, N_LOGO_FOLDS + 1)]
        else:
            names = [f"{args.arch}_fold_{k}.pt" for k in range(1, N_LOGO_FOLDS + 1)]
        checkpoint_paths = [models_dir / name for name in names]
    else:
        checkpoint_paths = [Path(args.checkpoint)]

    if weights is None:
        weights = [1.0 / len(checkpoint_paths)] * len(checkpoint_paths)  # equal weighting

    missing = [str(p) for p in checkpoint_paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing checkpoint file(s):\n  " + "\n  ".join(missing))

    sum_w = sum(weights)
    return checkpoint_paths, [w / sum_w for w in weights]


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Run possession inference on the competition match.")
    parser.add_argument("--checkpoint", type=str, default="models/bi_gru_full.pt",
                        help="Path to single model checkpoint (used if --checkpoints and --ensemble are not set).")
    parser.add_argument("--checkpoints", nargs="+", type=str, default=None,
                        help="List of model checkpoint paths for heterogeneous ensemble (.pt and/or .joblib).")
    parser.add_argument("--weights", nargs="+", type=float, default=None,
                        help="Optional ensemble weights for --checkpoints (defaults to equal weighting).")
    parser.add_argument("--ensemble", action="store_true",
                        help="Ensemble across the 3 LOGO fold checkpoints.")
    parser.add_argument("--arch", type=str, default="bi_gru",
                        help="Architecture family when using --ensemble ('bi_gru', 'tcn_1d', 'lgbm').")
    parser.add_argument("--models-dir", type=str, default="models",
                        help="Directory containing fold checkpoints when --ensemble is set.")
    parser.add_argument("--hybrid-dead-ball", action="store_true",
                        help="Enable Physics + ML Hybrid Dead-Ball fusion combining kinematic detection with neural emissions.")
    parser.add_argument("--dead-ball-boost", type=float, default=1.5,
                        help="Log-probability boost for Dead Ball (D) frames identified by kinematic detector (default: 1.5).")
    parser.add_argument("--output", type=str, default="results/lamasia_helvetia_bigru.csv",
                        help="Output submission CSV path.")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Match assets root (defaults to MTAH_DATA_DIR or ./data).")
    parser.add_argument("--seq-len", type=int, default=250, help="Sliding window sequence length.")
    parser.add_argument("--stride", type=int, default=125, help="Sliding window stride.")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size.")
    parser.add_argument("--trans-penalty-ha", type=float, default=3.0, help="Viterbi H <-> A transition penalty.")
    parser.add_argument("--trans-penalty-dead", type=float, default=2.0, help="Viterbi live <-> D transition penalty.")
    parser.add_argument("--abstain-threshold", type=float, default=0.45,
                        help="Confidence threshold for selective abstention ('X').")
    parser.add_argument("--no-abstain", action="store_true",
                        help="Disable selective abstention ('X') and emit only H, A, D.")
    parser.add_argument("--device", type=str, default=None, help="'cuda', 'cpu', or auto-detect.")
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject invalid or contradictory CLI values up front with a readable message."""
    args.arch = args.arch.lower().strip()
    if args.arch not in ARCH_ALIASES:
        parser.error(f"--arch '{args.arch}' is not supported; choose one of {sorted(ARCH_ALIASES)}")
    if args.checkpoints and args.ensemble:
        parser.error("--checkpoints and --ensemble are mutually exclusive")
    if args.weights is not None:
        if not args.checkpoints:
            parser.error("--weights requires --checkpoints")
        if len(args.weights) != len(args.checkpoints):
            parser.error(
                f"number of --weights ({len(args.weights)}) must match "
                f"number of --checkpoints ({len(args.checkpoints)})"
            )
        if any(not math.isfinite(w) or w < 0.0 for w in args.weights):
            parser.error("--weights must be finite and non-negative")
        if sum(args.weights) <= 0.0:
            parser.error("--weights must sum to a positive value")
    try:
        validate_window_args(args.seq_len, args.stride, args.batch_size)
    except ValueError as error:
        parser.error(str(error))
    for flag, value in (("--trans-penalty-ha", args.trans_penalty_ha),
                        ("--trans-penalty-dead", args.trans_penalty_dead),
                        ("--dead-ball-boost", args.dead_ball_boost)):
        if not math.isfinite(value) or value < 0.0:
            parser.error(f"{flag} must be a finite non-negative number, got {value}")
    if not 0.0 <= args.abstain_threshold <= 1.0:
        parser.error(f"--abstain-threshold must lie in [0, 1], got {args.abstain_threshold}")


def main() -> None:
    """Parse CLI arguments, run inference on the competition match and write the submission."""
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_args(parser, args)

    # Fail on missing checkpoints before the (slow) tracking ingestion
    try:
        checkpoint_paths, norm_weights = resolve_checkpoints(args)
    except FileNotFoundError as error:
        parser.error(str(error))

    seed_everything(INFERENCE_SEED)
    try:
        device = resolve_device(args.device)
    except ValueError as error:
        parser.error(str(error))

    # 1. Resolve match asset paths
    root_data = args.data_dir or DATA_DIR
    print(f"Loading match assets for {GAME} from: {root_data}")
    raw_path, meta_path, _ = require(["atd_raw", "atd_meta", "smart_tagging"], game=GAME, data_dir=root_data)
    pitch = GAMES[GAME]["pitch"]
    n_expected = GAMES[GAME]["n_video_frames"]
    fps = GAMES[GAME]["fps"]
    cfg = PipelineConfig()

    # 2. Ingest tracking data & reconstruct ball trajectory
    print(f"Ingesting raw ATD tracking ({n_expected} frames)...")
    tracking_data = load_raw_tracking_data(
        raw_path=Path(raw_path),
        meta_path=Path(meta_path),
        n_expected_frames=n_expected,
        pitch=pitch,
    )

    print("Reconstructing ball trajectory and identifying occlusion blackouts...")
    ball_res = reconstruct_ball_trajectory(
        tracking_data.ball_xy,
        pitch=pitch,
        max_gap_frames=cfg.max_ball_gap_frames,
        max_physical_speed_mps=cfg.max_ball_speed_mps,
        fps=fps,
        savgol_window=cfg.savgol_window,
        savgol_polyorder=cfg.savgol_polyorder,
    )

    # 3. Ingest Smart Tagging stoppages
    print("Ingesting Smart Tagging stoppage events...")
    smart_events = load_smart_tagging_events(game=GAME, data_dir=root_data)
    is_stoppage_tag = build_stoppage_tag_mask(smart_events, n_expected).astype(np.float32)

    # 4. Extract canonical 14-dimensional features
    print("Extracting canonical 14-dimensional feature representations...")
    features = extract_frame_features_from_tracking(
        tracking_data=tracking_data,
        is_stoppage_tag=is_stoppage_tag,
        ball_tracking_res=ball_res,
    )
    print(f"Extracted feature matrix shape: {features.shape} (dtype: {features.dtype})")

    # 5. Load checkpoints (heterogeneous ensemble)
    print(f"\nLoading {len(checkpoint_paths)} checkpoint(s):")
    models = []
    for p, w in zip(checkpoint_paths, norm_weights):
        m = load_model_from_checkpoint(p, device)
        models.append(m)
        print(f"  - {p} (architecture: {m.__class__.__name__}, weight: {w:.3f})")

    # 6. Physical Dead-Ball Detection (Hybrid Fusion)
    half_blocks = half_video_blocks(GAME)
    is_dead_ball = None
    if args.hybrid_dead_ball:
        print("\nDetecting physical Dead Ball ('D') intervals via Smart Tagging and player kinematics...")
        dead_ball_result = detect_dead_ball(
            n_frames=n_expected,
            mean_player_speed=tracking_data.mean_player_speed,
            p80_player_speed=tracking_data.p80_player_speed,
            ball_xy=ball_res.ball_xy,
            is_ball_uncertain=ball_res.is_ball_uncertain,
            ball_speed=ball_res.ball_speed,
            player_xy=tracking_data.player_xy,
            smart_tagging_events=smart_events,
            half_blocks=half_blocks,
            pitch=pitch,
            jogging_speed_threshold_mps=cfg.jogging_speed_threshold_mps,
            kinematic_stoppage_frames=cfg.kinematic_stoppage_frames,
            dilation_window_frames=cfg.dilation_window_frames,
            min_stoppage_duration_frames=cfg.min_stoppage_duration_frames,
            stationary_ball_speed_mps=cfg.stationary_ball_speed_mps,
            abandoned_ball_distance_m=cfg.abandoned_ball_distance_m,
            standalone_stoppage_frames=cfg.standalone_stoppage_frames,
            standalone_p80_threshold_mps=cfg.standalone_p80_threshold_mps,
            fps=fps,
        )
        is_dead_ball = dead_ball_result.is_dead_ball
        n_dead_phys = int(np.sum(is_dead_ball))
        print(f"Physical dead-ball detector flagged {n_dead_phys}/{n_expected} frames ({100.0 * n_dead_phys / n_expected:.1f}%).")

    # 7. Independent half-by-half decoding; everything outside the halves stays Dead Ball
    final_codes = np.full(n_expected, DEAD_BALL_CODE, dtype=np.int8)

    for half_num, first_frame, last_frame in half_blocks:
        h_start = max(0, first_frame)
        h_stop = min(n_expected, last_frame + 1)  # exclusive
        if h_start >= h_stop:
            continue
        half = slice(h_start, h_stop)

        print(f"\nProcessing Half {half_num} (frames {h_start}..{h_stop - 1}, span: {h_stop - h_start})...")
        h_features = features[half]
        # Same visibility the models were fed (feature 3), so abstention matches their input
        h_ball_vis = h_features[:, BALL_VISIBLE_INDEX] > 0.5

        # Predict weighted average probabilities across all models
        h_probs = predict_half_probabilities(
            models=models,
            weights=norm_weights,
            features=h_features,
            device=device,
            seq_len=args.seq_len,
            stride=args.stride,
            batch_size=args.batch_size,
        )

        # Back to (finite) log-scores, with optional Physics + ML Hybrid Dead-Ball Fusion
        h_emissions = fuse_dead_ball_evidence(
            h_probs,
            is_dead=is_dead_ball[half] if is_dead_ball is not None else None,
            dead_ball_boost=args.dead_ball_boost,
        )

        # Decode via Viterbi
        final_codes[half] = viterbi_decode(
            emissions=h_emissions,
            is_ball_visible=h_ball_vis,
            trans_penalty_ha=args.trans_penalty_ha,
            trans_penalty_dead=args.trans_penalty_dead,
            abstain_threshold=args.abstain_threshold,
            enable_abstention=not args.no_abstain,
            emission_type="logits",
        )

    # 8. Convert codes to string labels and validate submission contract
    # codes_to_labels raises on any code outside {0, 1, 2, 3}, so no blank/invalid label can be written
    label_strings = codes_to_labels(final_codes)
    df_submission = pd.DataFrame({
        "frame": np.arange(n_expected, dtype=np.int64),
        "label": label_strings,
    })

    print(f"\nValidating submission against official contract ({n_expected} rows)...")
    reason = pm.validate_submission(df_submission, n_expected)
    if reason is not None:
        raise RuntimeError(f"Submission validation failed: {reason}")
    print("Format validation PASSED!")

    # 9. Football analytics summary
    n_spells, mean_dur = describe(final_codes, fps)
    print(f"\nSubmission claims {n_spells} possessions with mean duration {mean_dur:.1f} s")
    print("Label distribution:")
    for idx, name in enumerate(pm.LABELS):
        count = int(np.sum(final_codes == idx))
        print(f"  {name}: {count:>7} frames ({100.0 * count / n_expected:5.1f}%)")

    # 10. Write submission CSV
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_submission.to_csv(out_path, index=False)
    print(f"\nSuccessfully saved submission to: {out_path}")


if __name__ == "__main__":
    main()
