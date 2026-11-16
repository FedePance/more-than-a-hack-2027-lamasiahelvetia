"""External LOGO Ensemble Weight Tuning CLI.

Tunes blending weights between BiGRU and TCN-1D sequence models across the 3
Leave-One-Game-Out (LOGO) cross-validation folds on Metrica Sample Games 1-3.

Workflow:
1. Loads held-out validation games (Sample_Game_1, Sample_Game_2, Sample_Game_3).
2. Applies deterministic 41% synthetic ball blackout mask to replicate competition conditions.
3. For each fold k in [1, 2, 3]:
   - Loads models/bi_gru_fold_{k}.pt and models/tcn_1d_fold_{k}.pt.
   - Computes and caches per-frame (n_frames, 3) softmax probabilities.
4. Evaluates candidate weight grid:
   [(1.0, 0.0), (0.8, 0.2), (0.65, 0.35), (0.5, 0.5), (0.35, 0.65), (0.2, 0.8), (0.0, 1.0)].
5. Runs Viterbi decoding with selective abstention ('X') and scores via official metrics
   (Spell F1, Macro F1).
6. Outputs formatted comparison table and ready-to-run infer_match.py command.
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

# Add project root and evaluation directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))

import possession_metrics as pm

from src.ml_pipeline.dataset_open import (
    OpenGameData,
    generate_ball_dropout_mask,
    load_open_game_data,
)
from src.ml_pipeline.features import N_FEATURES
from src.ml_pipeline.models import build_model
from src.ml_pipeline.viterbi import viterbi_decode

CANDIDATE_WEIGHTS = [
    (1.0, 0.0),
    (0.8, 0.2),
    (0.65, 0.35),
    (0.5, 0.5),
    (0.35, 0.65),
    (0.2, 0.8),
    (0.0, 1.0),
]


def seed_everything(seed: int = 42) -> None:
    """Set random seed across all libraries for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_game(
    game_id: int,
    cache_dir: Path | str,
    data_dir: Path | str | None = None,
) -> OpenGameData:
    """Load OpenGameData container for Sample_Game_{game_id}, checking cache first."""
    c_dir = Path(cache_dir)
    game_str = f"Sample_Game_{game_id}"
    cache_file = c_dir / f"{game_str}.npz"
    if cache_file.is_file():
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
        except Exception:
            pass  # Fall back to raw loading

    return load_open_game_data(game_id, data_dir=data_dir)


def load_checkpoint(
    checkpoint_path: Path | str,
    device: torch.device,
    expected_arch: str,
) -> nn.Module:
    """Load model architecture and weights from saved checkpoint file (.pt).

    Args:
        checkpoint_path: Path to checkpoint .pt file.
        device: Torch execution device.
        expected_arch: Expected architecture name ('bi_gru' or 'tcn_1d').

    Returns:
        Evaluatable nn.Module loaded on device.
    """
    p = Path(checkpoint_path)
    if not p.is_file():
        raise FileNotFoundError(
            f"Checkpoint file not found: {p}\n"
            f"Please verify model training completed via:\n"
            f"  python src/ml_pipeline/train_logo.py --arch {expected_arch}"
        )

    ckpt = torch.load(p, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        arch = ckpt.get("arch", expected_arch)
        hidden_size = ckpt.get("hidden_size", 96)
        num_layers = ckpt.get("num_layers", 2)
        dropout = ckpt.get("dropout", 0.25)
        input_dim = ckpt.get("input_dim", N_FEATURES)
        num_classes = ckpt.get("num_classes", 3)
    elif isinstance(ckpt, dict):
        state_dict = ckpt
        arch = expected_arch
        hidden_size = 96
        num_layers = 2
        dropout = 0.25
        input_dim = N_FEATURES
        num_classes = 3
    else:
        raise ValueError(f"Unrecognized checkpoint format in {p}")

    model = build_model(
        arch=arch,
        input_dim=input_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        num_classes=num_classes,
    )
    model = model.to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def prepare_masked_game_features(
    game: OpenGameData,
    target_ratio: float = 0.41,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply deterministic 41% ball dropout mask and reset ball feature channels.

    Args:
        game: OpenGameData match container.
        target_ratio: Target occlusion fraction (0.41).
        seed: Deterministic random seed for dropout mask.

    Returns:
        Tuple of (val_features, ball_vis) where:
        - val_features: (n_frames, 14) feature array with masked ball channels.
        - ball_vis: (n_frames,) boolean mask indicating effective ball visibility.
    """
    n_frames = game.n_frames
    val_dropout_mask = generate_ball_dropout_mask(n_frames, target_ratio=target_ratio, seed=seed)
    ball_vis = game.is_ball_visible.copy() & ~val_dropout_mask

    val_features = game.features.copy()
    masked = ~ball_vis
    val_features[masked, 0] = 50.0  # min_dist_home
    val_features[masked, 1] = 50.0  # min_dist_away
    val_features[masked, 2] = 0.0   # dist_diff
    val_features[masked, 3] = 0.0   # is_ball_visible
    val_features[masked, 4] = 0.0   # ball_speed
    val_features[masked, 5] = 0.0   # ball_x_norm
    val_features[masked, 6] = 0.0   # ball_y_norm
    val_features[masked, 7] = 0.0   # closest_home_speed
    val_features[masked, 8] = 0.0   # closest_away_speed

    return val_features, ball_vis


def predict_model_probabilities(
    model: nn.Module,
    features: np.ndarray,
    device: torch.device,
    seq_len: int = 250,
    stride: int = 125,
    batch_size: int = 64,
) -> np.ndarray:
    """Run model inference across full match frames using overlapping window averaging.

    Args:
        model: Trained PyTorch sequence model in eval mode.
        features: Frame features array of shape (n_frames, n_features).
        device: Torch execution device.
        seq_len: Slicing sequence length in frames.
        stride: Slicing step in frames.
        batch_size: Window batch size.

    Returns:
        Per-frame softmax probabilities of shape (n_frames, 3).
    """
    model.eval()
    n_frames = len(features)
    accum_logits = np.zeros((n_frames, 3), dtype=np.float32)
    counts = np.zeros(n_frames, dtype=np.float32)

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

    # Numerically stable softmax probabilities
    shift = avg_logits - np.max(avg_logits, axis=-1, keepdims=True)
    exp_vals = np.exp(shift)
    probs = (exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)).astype(np.float32)
    return probs


def main():
    parser = argparse.ArgumentParser(
        description="Tune external LOGO ensemble blending weights between BiGRU and TCN-1D."
    )
    parser.add_argument(
        "--models-dir", type=str, default="models",
        help="Directory containing trained fold checkpoints (default: 'models')."
    )
    parser.add_argument(
        "--cache-dir", type=str, default="external_open/.cache",
        help="Directory containing cached game .npz files (default: 'external_open/.cache')."
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Optional root folder for raw open data if cache is missing."
    )
    parser.add_argument(
        "--window-size", "--seq-len", dest="window_size", type=int, default=250,
        help="Sliding window sequence length in frames (default: 250)."
    )
    parser.add_argument(
        "--stride", type=int, default=125,
        help="Sliding window stride in frames (default: 125)."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Inference batch size for sliding windows (default: 64)."
    )
    parser.add_argument(
        "--trans-penalty-ha", type=float, default=3.0,
        help="Viterbi H <-> A transition penalty (default: 3.0)."
    )
    parser.add_argument(
        "--trans-penalty-dead", type=float, default=2.0,
        help="Viterbi live <-> D transition penalty (default: 2.0)."
    )
    parser.add_argument(
        "--abstain-threshold", type=float, default=0.45,
        help="Confidence threshold for selective abstention 'X' (default: 0.45)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for deterministic execution (default: 42)."
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="'cuda', 'cpu', or auto-detect if None."
    )
    args = parser.parse_args()

    seed_everything(args.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    if device.type == "cpu":
        n_threads = min(4, os.cpu_count() or 4)
        torch.set_num_threads(n_threads)
        print(f"Configured PyTorch CPU threads: {n_threads}", flush=True)

    models_dir = Path(args.models_dir)
    cache_dir = Path(args.cache_dir)

    print(f"\n{'='*90}", flush=True)
    print("STEP 1: INGESTING HELD-OUT VALIDATION GAMES & PRECOMPUTING MODEL PROBABILITIES", flush=True)
    print(f"{'='*90}", flush=True)

    fold_data = {}

    for k in [1, 2, 3]:
        game_id_str = f"Sample_Game_{k}"
        print(f"\n--- Fold {k}: Held-out Validation Game {game_id_str} ---", flush=True)

        # 1. Load held-out game
        game = load_game(game_id=k, cache_dir=cache_dir, data_dir=args.data_dir)
        print(
            f"Loaded {game.game_id}: {game.n_frames} frames "
            f"({game.n_frames / game.fps / 60.0:.1f} mins)",
            flush=True,
        )

        # 2. Mask features under 41% deterministic synthetic dropout
        val_features, ball_vis = prepare_masked_game_features(game, target_ratio=0.41, seed=args.seed)
        n_masked = int((~ball_vis).sum())
        print(
            f"Applied 41% synthetic ball blackout: {n_masked}/{game.n_frames} frames occluded "
            f"({100.0 * n_masked / game.n_frames:.1f}%)",
            flush=True,
        )

        # 3. Load BiGRU fold checkpoint and predict probabilities
        bigru_ckpt = models_dir / f"bi_gru_fold_{k}.pt"
        print(f"Loading BiGRU fold checkpoint: {bigru_ckpt}...", flush=True)
        bigru_model = load_checkpoint(bigru_ckpt, device, expected_arch="bi_gru")
        probs_bigru = predict_model_probabilities(
            model=bigru_model,
            features=val_features,
            device=device,
            seq_len=args.window_size,
            stride=args.stride,
            batch_size=args.batch_size,
        )
        print(f"BiGRU probability matrix computed: shape {probs_bigru.shape}", flush=True)

        # 4. Load TCN-1D fold checkpoint and predict probabilities
        tcn_ckpt = models_dir / f"tcn_1d_fold_{k}.pt"
        print(f"Loading TCN-1D fold checkpoint: {tcn_ckpt}...", flush=True)
        tcn_model = load_checkpoint(tcn_ckpt, device, expected_arch="tcn_1d")
        probs_tcn = predict_model_probabilities(
            model=tcn_model,
            features=val_features,
            device=device,
            seq_len=args.window_size,
            stride=args.stride,
            batch_size=args.batch_size,
        )
        print(f"TCN-1D probability matrix computed: shape {probs_tcn.shape}", flush=True)

        fold_data[k] = {
            "game_id": game_id_str,
            "gt_labels": game.gt_labels,
            "ball_vis": ball_vis,
            "probs_bigru": probs_bigru,
            "probs_tcn": probs_tcn,
        }

    print(f"\n{'='*90}", flush=True)
    print("STEP 2: EVALUATING ENSEMBLE WEIGHT GRID VIA VITERBI DECODING", flush=True)
    print(f"{'='*90}", flush=True)

    results = []

    for w_bigru, w_tcn in CANDIDATE_WEIGHTS:
        fold_spell_f1s = []
        fold_macro_f1s = []

        for k in [1, 2, 3]:
            fd = fold_data[k]
            # Convex blend of softmax probabilities
            blended_probs = w_bigru * fd["probs_bigru"] + w_tcn * fd["probs_tcn"]

            # Numba-accelerated Viterbi decoding with selective abstention ('X')
            pred_codes = viterbi_decode(
                emissions=blended_probs,
                is_ball_visible=fd["ball_vis"],
                trans_penalty_ha=args.trans_penalty_ha,
                trans_penalty_dead=args.trans_penalty_dead,
                abstain_threshold=args.abstain_threshold,
                enable_abstention=True,
            )

            # Official competition possession metrics evaluation
            metrics = pm.evaluate(fd["gt_labels"], pred_codes)
            fold_spell_f1s.append(metrics["spell_f1_mean"])
            fold_macro_f1s.append(metrics["macro_f1"])

        mean_spell_f1 = float(np.mean(fold_spell_f1s))
        std_spell_f1 = float(np.std(fold_spell_f1s))
        mean_macro_f1 = float(np.mean(fold_macro_f1s))

        results.append({
            "w_bigru": w_bigru,
            "w_tcn": w_tcn,
            "fold_scores": fold_spell_f1s,
            "fold_macros": fold_macro_f1s,
            "mean_spell_f1": mean_spell_f1,
            "std_spell_f1": std_spell_f1,
            "mean_macro_f1": mean_macro_f1,
        })

    # Identify optimal configuration maximizing Mean CV Spell F1 (tie-break on Macro F1)
    best_res = max(results, key=lambda r: (r["mean_spell_f1"], r["mean_macro_f1"]))

    # Print comparison table
    print("\n" + "=" * 104)
    print(
        f"{'Weights (BiGRU / TCN)':<24} | {'Fold 1':<8} | {'Fold 2':<8} | {'Fold 3':<8} | "
        f"{'Mean CV Spell F1':<22} | {'Mean CV Macro F1':<16}"
    )
    print("-" * 104)

    for r in results:
        w_str = f"{r['w_bigru']:.2f} / {r['w_tcn']:.2f}"
        if r["w_bigru"] == 1.0 and r["w_tcn"] == 0.0:
            w_str += " (Pure BiGRU)"
        elif r["w_bigru"] == 0.0 and r["w_tcn"] == 1.0:
            w_str += " (Pure TCN)"

        f1_str = f"{r['fold_scores'][0]:.4f}"
        f2_str = f"{r['fold_scores'][1]:.4f}"
        f3_str = f"{r['fold_scores'][2]:.4f}"
        mean_spell_str = f"{r['mean_spell_f1']:.4f} (+/- {r['std_spell_f1']:.4f})"
        mean_macro_str = f"{r['mean_macro_f1']:.4f}"

        marker = " [*BEST]" if (r["w_bigru"] == best_res["w_bigru"] and r["w_tcn"] == best_res["w_tcn"]) else ""
        print(
            f"{w_str:<24} | {f1_str:<8} | {f2_str:<8} | {f3_str:<8} | "
            f"{mean_spell_str:<22} | {mean_macro_str:<16}{marker}"
        )

    print("=" * 104)

    # Print optimal summary and ready-to-run inference command
    opt_w_bg = best_res["w_bigru"]
    opt_w_tcn = best_res["w_tcn"]
    print(f"\n{'='*104}")
    print("OPTIMAL EXTERNAL ENSEMBLE CONFIGURATION")
    print(f"{'='*104}")
    print(f"Optimal Weights : BiGRU = {opt_w_bg:.2f}, TCN-1D = {opt_w_tcn:.2f}")
    print(f"Mean CV Spell F1: {best_res['mean_spell_f1']:.4f} (+/- {best_res['std_spell_f1']:.4f})")
    print(f"Mean CV Macro F1: {best_res['mean_macro_f1']:.4f}")
    print(f"Fold 1 Spell F1 : {best_res['fold_scores'][0]:.4f} (Macro F1: {best_res['fold_macros'][0]:.4f})")
    print(f"Fold 2 Spell F1 : {best_res['fold_scores'][1]:.4f} (Macro F1: {best_res['fold_macros'][1]:.4f})")
    print(f"Fold 3 Spell F1 : {best_res['fold_scores'][2]:.4f} (Macro F1: {best_res['fold_macros'][2]:.4f})")
    print(f"{'-'*104}")
    print("READY-TO-RUN INFERENCE COMMAND (FULL-FIT ENSEMBLE):")
    print(
        "python src/ml_pipeline/infer_match.py \\\n"
        f"  --checkpoints models/bi_gru_full.pt models/tcn_1d_full.pt \\\n"
        f"  --weights {opt_w_bg:.2f} {opt_w_tcn:.2f} \\\n"
        "  --output results/lamasia_helvetia_ensemble.csv"
    )
    print(f"{'-'*104}")
    print("READY-TO-RUN INFERENCE COMMAND (6-FOLD HETEROGENEOUS ENSEMBLE):")
    bg_sub_w = opt_w_bg / 3.0
    tcn_sub_w = opt_w_tcn / 3.0
    print(
        "python src/ml_pipeline/infer_match.py \\\n"
        "  --checkpoints \\\n"
        "    models/bi_gru_fold_1.pt models/bi_gru_fold_2.pt models/bi_gru_fold_3.pt \\\n"
        "    models/tcn_1d_fold_1.pt models/tcn_1d_fold_2.pt models/tcn_1d_fold_3.pt \\\n"
        f"  --weights {bg_sub_w:.4f} {bg_sub_w:.4f} {bg_sub_w:.4f} {tcn_sub_w:.4f} {tcn_sub_w:.4f} {tcn_sub_w:.4f} \\\n"
        "  --output results/lamasia_helvetia_ensemble_folds.csv"
    )
    print(f"{'='*104}\n")


if __name__ == "__main__":
    main()
