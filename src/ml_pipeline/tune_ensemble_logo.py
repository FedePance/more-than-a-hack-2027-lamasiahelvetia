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
import sys
from pathlib import Path

import numpy as np

# Add project root and evaluation directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))

import possession_metrics as pm

from src.ml_pipeline.dataset_open import (
    OpenGameData,
    load_cached_game,
    load_open_game_data,
    prepare_masked_game_features,
)
from src.ml_pipeline.inference import (
    load_model_from_checkpoint,
    predict_sliding_window_probabilities,
    resolve_device,
    seed_everything,
    validate_window_args,
)
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
FOLD_IDS = (1, 2, 3)
VAL_DROPOUT_RATIO = 0.41


def load_game(
    game_id: int,
    cache_dir: Path | str,
    data_dir: Path | str | None = None,
) -> OpenGameData:
    """Load OpenGameData container for Sample_Game_{game_id}, checking cache_dir first."""
    game_str = f"Sample_Game_{game_id}"
    cached_game = load_cached_game(Path(cache_dir) / f"{game_str}.npz", game_str)
    if cached_game is not None:
        return cached_game
    return load_open_game_data(game_id, data_dir=data_dir)


def main() -> None:
    """Parse CLI arguments and evaluate the BiGRU / TCN-1D blending weight grid."""
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

    try:
        validate_window_args(args.window_size, args.stride, args.batch_size)
    except ValueError as error:
        parser.error(str(error))
    if args.trans_penalty_ha < 0 or args.trans_penalty_dead < 0:
        parser.error("--trans-penalty-ha and --trans-penalty-dead must be >= 0")
    if not 0.0 <= args.abstain_threshold <= 1.0:
        parser.error("--abstain-threshold must lie in [0, 1]")

    models_dir = Path(args.models_dir)
    cache_dir = Path(args.cache_dir)

    # Fail before any heavy work if a fold checkpoint is missing
    fold_checkpoints = {
        k: {"bi_gru": models_dir / f"bi_gru_fold_{k}.pt", "tcn_1d": models_dir / f"tcn_1d_fold_{k}.pt"}
        for k in FOLD_IDS
    }
    missing = [str(p) for ckpts in fold_checkpoints.values() for p in ckpts.values() if not p.is_file()]
    if missing:
        parser.error(
            "missing fold checkpoint(s):\n  " + "\n  ".join(missing)
            + "\nTrain them with: python src/ml_pipeline/train_logo.py --mode logo --arch {bi_gru,tcn_1d}"
        )

    seed_everything(args.seed)
    try:
        device = resolve_device(args.device)
    except ValueError as error:
        parser.error(str(error))

    print(f"\n{'='*90}", flush=True)
    print("STEP 1: INGESTING HELD-OUT VALIDATION GAMES & PRECOMPUTING MODEL PROBABILITIES", flush=True)
    print(f"{'='*90}", flush=True)

    fold_data = {}

    for k in FOLD_IDS:
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
        val_features, ball_vis = prepare_masked_game_features(
            game, target_ratio=VAL_DROPOUT_RATIO, seed=args.seed
        )
        n_masked = int((~ball_vis).sum())
        print(
            f"Applied 41% synthetic ball blackout: {n_masked}/{game.n_frames} frames occluded "
            f"({100.0 * n_masked / game.n_frames:.1f}%)",
            flush=True,
        )

        # 3. Load each fold checkpoint and predict per-frame probabilities
        fold_probs = {}
        for arch, display_name in (("bi_gru", "BiGRU"), ("tcn_1d", "TCN-1D")):
            ckpt_path = fold_checkpoints[k][arch]
            print(f"Loading {display_name} fold checkpoint: {ckpt_path}...", flush=True)
            model = load_model_from_checkpoint(ckpt_path, device, default_arch=arch)
            fold_probs[arch] = predict_sliding_window_probabilities(
                model=model,
                features=val_features,
                device=device,
                seq_len=args.window_size,
                stride=args.stride,
                batch_size=args.batch_size,
            )
            print(f"{display_name} probability matrix computed: shape {fold_probs[arch].shape}", flush=True)

        fold_data[k] = {
            "game_id": game_id_str,
            "gt_labels": game.gt_labels,
            "ball_vis": ball_vis,
            "probs_bigru": fold_probs["bi_gru"],
            "probs_tcn": fold_probs["tcn_1d"],
        }

    print(f"\n{'='*90}", flush=True)
    print("STEP 2: EVALUATING ENSEMBLE WEIGHT GRID VIA VITERBI DECODING", flush=True)
    print(f"{'='*90}", flush=True)

    results = []

    for w_bigru, w_tcn in CANDIDATE_WEIGHTS:
        fold_spell_f1s = []
        fold_macro_f1s = []

        for k in FOLD_IDS:
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
                emission_type="probs",
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
