"""Inference CLI for competition match (BAR-ATH).

Loads match assets via games.py, src/data_loader.py, src/ball_tracking.py,
and starter/load_smart_tagging.py, extracts canonical 14-dimensional features,
runs model inference (single checkpoint, heterogeneous ensemble, or LOGO 3-fold ensemble),
optionally fuses physical dead-ball detections with neural emission distributions,
decodes each half independently via Viterbi decoding, validates format,
and writes submission to results/lamasia_helvetia_*.csv.
"""

import argparse
import os
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
from src.dead_ball import (
    STOPPAGE_CODES,
    STOPPAGE_SET_PIECE_TYPES,
    detect_dead_ball,
)
from src.ml_pipeline.features import (
    N_FEATURES,
    extract_frame_features_from_tracking,
)
from src.ml_pipeline.models import build_model
from src.ml_pipeline.viterbi import codes_to_labels, viterbi_decode
from starter.load_smart_tagging import load as load_smart_tagging_events


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> nn.Module:
    """Load model architecture and weights from saved checkpoint file.

    Inspects ckpt['arch'] and saved hyperparameters, defaulting gracefully
    if older checkpoint dictionaries omitted them.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    arch = ckpt.get("arch")
    if not arch:
        p_str = str(checkpoint_path).lower()
        if "tcn" in p_str:
            arch = "tcn_1d"
        else:
            arch = "bi_gru"

    hidden_size = ckpt.get("hidden_size", 96)
    num_layers = ckpt.get("num_layers", 2)
    dropout = ckpt.get("dropout", 0.25)
    input_dim = ckpt.get("input_dim", N_FEATURES)
    num_classes = ckpt.get("num_classes", 3)

    model = build_model(
        arch=arch,
        input_dim=input_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        num_classes=num_classes,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def predict_half_probabilities(
    models: list[nn.Module],
    weights: list[float] | np.ndarray,
    features: np.ndarray,
    device: torch.device,
    seq_len: int = 250,
    stride: int = 125,
    batch_size: int = 64,
) -> np.ndarray:
    """Predict probabilities across a half, computing weighted average across models."""
    n_frames = len(features)
    weighted_probs = np.zeros((n_frames, 3), dtype=np.float32)

    windows = []
    spans = []
    max_start = n_frames - seq_len
    starts = list(range(0, max_start + 1, stride))
    if not starts or starts[-1] < max_start:
        starts.append(max_start)

    for start in starts:
        end = start + seq_len
        windows.append(features[start:end])
        spans.append((start, end))

    for model, w in zip(models, weights):
        model.eval()
        accum_logits = np.zeros((n_frames, 3), dtype=np.float32)
        counts = np.zeros(n_frames, dtype=np.float32)

        with torch.no_grad():
            for b_start in range(0, len(windows), batch_size):
                b_end = min(len(windows), b_start + batch_size)
                batch_data = np.stack(windows[b_start:b_end], axis=0)
                tensor_x = torch.from_numpy(batch_data).float().to(device)
                batch_logits = model(tensor_x).cpu().numpy()

                for i, (s, e) in enumerate(spans[b_start:b_end]):
                    accum_logits[s:e] += batch_logits[i]
                    counts[s:e] += 1.0

        counts = np.maximum(counts, 1.0)[:, None]
        avg_logits = accum_logits / counts

        # Compute per-frame softmax probabilities
        shift = avg_logits - np.max(avg_logits, axis=-1, keepdims=True)
        exp_vals = np.exp(shift)
        m_probs = exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)

        weighted_probs += float(w) * m_probs

    return weighted_probs


def main():
    parser = argparse.ArgumentParser(description="Run possession inference on the competition match.")
    parser.add_argument("--checkpoint", type=str, default="models/bi_gru_full.pt",
                        help="Path to single model checkpoint (used if --checkpoints and --ensemble are not set).")
    parser.add_argument("--checkpoints", nargs="+", type=str, default=None,
                        help="List of model checkpoint paths for heterogeneous ensemble.")
    parser.add_argument("--weights", nargs="+", type=float, default=None,
                        help="Optional ensemble weights for --checkpoints (defaults to equal weighting).")
    parser.add_argument("--ensemble", action="store_true",
                        help="Ensemble across the 3 LOGO fold checkpoints (bi_gru_fold_1/2/3.pt).")
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
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    if device.type == "cpu":
        n_threads = min(4, os.cpu_count() or 4)
        torch.set_num_threads(n_threads)
        print(f"Configured PyTorch CPU threads: {n_threads}", flush=True)

    # 1. Resolve match asset paths
    root_data = args.data_dir or DATA_DIR
    print(f"Loading match assets for {GAME} from: {root_data}")
    raw_path, meta_path = require(["atd_raw", "atd_meta"], game=GAME, data_dir=root_data)
    pitch = GAMES[GAME]["pitch"]
    n_expected = GAMES[GAME]["n_video_frames"]

    # 2. Ingest tracking data & reconstruct ball trajectory
    print(f"Ingesting raw ATD tracking ({n_expected} frames)...")
    tracking_data = load_raw_tracking_data(
        raw_path=Path(raw_path),
        meta_path=Path(meta_path),
        n_expected_frames=n_expected,
        pitch=pitch,
    )

    print("Reconstructing ball trajectory and identifying occlusion blackouts...")
    ball_res = reconstruct_ball_trajectory(tracking_data.ball_xy, pitch=pitch)

    # 3. Ingest Smart Tagging stoppages
    print("Ingesting Smart Tagging stoppage events...")
    smart_events = load_smart_tagging_events(game=GAME, data_dir=root_data)
    is_stoppage_tag = np.zeros(n_expected, dtype=np.float32)
    for ev in smart_events:
        code = (ev.get("code") or "").strip().lower()
        sf = max(0, ev.get("start_frame", 0))
        ef = min(n_expected, ev.get("end_frame", 0))
        if sf >= ef:
            continue
        if code in STOPPAGE_CODES:
            is_stoppage_tag[sf:ef] = 1.0
        elif "set piece" in code:
            etype = (ev.get("tags", {}).get("Type") or "").strip().lower()
            if not etype or any(t in etype for t in STOPPAGE_SET_PIECE_TYPES):
                is_stoppage_tag[sf:ef] = 1.0

    # 4. Extract canonical 14-dimensional features
    print("Extracting canonical 14-dimensional feature representations...")
    features = extract_frame_features_from_tracking(
        tracking_data=tracking_data,
        is_stoppage_tag=is_stoppage_tag,
        ball_tracking_res=ball_res,
    )
    print(f"Extracted feature matrix shape: {features.shape} (dtype: {features.dtype})")

    # 5. Resolve checkpoint list & weights (Heterogeneous Ensemble)
    if args.checkpoints:
        checkpoint_paths = [Path(p) for p in args.checkpoints]
        if args.weights:
            if len(args.weights) != len(checkpoint_paths):
                raise ValueError(
                    f"Number of --weights ({len(args.weights)}) must match "
                    f"number of --checkpoints ({len(checkpoint_paths)})"
                )
            weights = [float(w) for w in args.weights]
        else:
            weights = [1.0 / len(checkpoint_paths)] * len(checkpoint_paths)
    elif args.ensemble:
        models_dir = Path(args.models_dir)
        checkpoint_paths = [
            models_dir / "bi_gru_fold_1.pt",
            models_dir / "bi_gru_fold_2.pt",
            models_dir / "bi_gru_fold_3.pt",
        ]
        weights = [1.0 / len(checkpoint_paths)] * len(checkpoint_paths)
    else:
        ckpt_path = Path(args.checkpoint)
        checkpoint_paths = [ckpt_path]
        weights = [1.0]

    # Verify all checkpoint files exist
    for p in checkpoint_paths:
        if not p.is_file():
            raise FileNotFoundError(f"Missing checkpoint file: {p}")

    # Normalize weights
    sum_w = sum(weights)
    if sum_w <= 0:
        raise ValueError("Ensemble weights must sum to a positive value.")
    norm_weights = [w / sum_w for w in weights]

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
        cfg = PipelineConfig()
        fps = GAMES[GAME]["fps"]
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

    # 7. Independent half-by-half decoding
    final_codes = np.full(n_expected, 2, dtype=np.int8)  # default Dead Ball (D = 2)

    for half_num, h_start, h_end in half_blocks:
        print(f"\nProcessing Half {half_num} (frames {h_start}..{h_end}, span: {h_end - h_start + 1})...")
        h_features = features[h_start : h_end + 1]
        h_ball_vis = ~ball_res.is_ball_uncertain[h_start : h_end + 1]

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

        # Convert back to log-probabilities
        h_emissions = np.log(np.clip(h_probs, 1e-7, 1.0)).astype(np.float32)

        # Physics + ML Hybrid Dead-Ball Fusion
        if args.hybrid_dead_ball and is_dead_ball is not None:
            h_is_dead = is_dead_ball[h_start : h_end + 1]
            h_emissions[h_is_dead, 2] += args.dead_ball_boost
            h_emissions[~h_is_dead, 2] -= 0.5 * args.dead_ball_boost

        # Decode via Viterbi
        decoded_half = viterbi_decode(
            emissions=h_emissions,
            is_ball_visible=h_ball_vis,
            trans_penalty_ha=args.trans_penalty_ha,
            trans_penalty_dead=args.trans_penalty_dead,
            abstain_threshold=args.abstain_threshold,
            enable_abstention=not args.no_abstain,
        )

        final_codes[h_start : h_end + 1] = decoded_half

    # Ensure all frames outside official halves are Dead Ball (D = 2)
    in_half = np.zeros(n_expected, dtype=bool)
    for _, s, e in half_blocks:
        in_half[s : e + 1] = True
    final_codes[~in_half] = 2

    # 8. Convert codes to string labels and validate submission contract
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
    fps = GAMES[GAME]["fps"]
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
