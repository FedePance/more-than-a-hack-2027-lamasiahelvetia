"""Leave-One-Game-Out (LOGO) Cross-Validation and Full Training CLI.

Supports:
1. --mode logo: 3-fold Leave-One-Game-Out cross validation across Metrica Sample Games 1-3.
   Trains on 2 games with fast on-the-fly augmentations and evaluates on the held-out game
   under deterministic 41% synthetic ball dropout using Numba-accelerated Viterbi decoding
   and official metrics (spell_f1_mean, macro_f1).
   Saves checkpoints to models/{arch}_fold_{k}.pt (or models/lgbm_fold_{k}.joblib for LightGBM).
2. --mode full: Trains on all 3 games combined and saves to models/{arch}_full.pt
   (or models/lgbm_full.joblib for LightGBM).
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root and evaluation directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))

import possession_metrics as pm

from src.ml_pipeline.dataset_open import (
    OpenGameData,
    OpenGameDataset,
    apply_domain_augmentations,
    generate_ball_dropout_mask,
    load_open_game_data,
    prepare_masked_game_features,
)
from src.ml_pipeline.features import N_FEATURES
from src.ml_pipeline.inference import (
    MAX_CPU_THREADS,
    N_CLASSES,
    predict_sliding_window_logits,
    resolve_device,
    seed_everything,
    validate_window_args,
)
from src.ml_pipeline.models import ARCH_ALIASES, PossessionLGBM, build_model
from src.ml_pipeline.viterbi import viterbi_decode

# Synthetic ball blackout applied to the held-out game, mirroring the competition feed
VAL_DROPOUT_RATIO = 0.41
VAL_DROPOUT_SEED = 42
N_AUGMENTED_VARIANTS = 2


def compute_class_weights(labels_list: list[np.ndarray], device: torch.device) -> torch.Tensor:
    """Compute balanced inverse class frequency weights for CrossEntropyLoss."""
    all_labels = np.concatenate(labels_list).astype(np.int64)
    counts = np.bincount(all_labels, minlength=N_CLASSES).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    total = float(len(all_labels))
    weights = total / (float(N_CLASSES) * counts)
    weights = weights / weights.mean()
    return torch.from_numpy(weights).float().to(device)


def temporal_smoothness_loss(logits: torch.Tensor) -> torch.Tensor:
    """Compute L2 penalty on consecutive frame logits to discourage high-frequency jitter.

    Args:
        logits: Tensor of shape (batch_size, seq_len, num_classes).

    Returns:
        Scalar mean squared difference between consecutive frames.
    """
    diffs = logits[:, 1:, :] - logits[:, :-1, :]
    return torch.mean(diffs * diffs)


def evaluate_game(
    model: nn.Module | PossessionLGBM,
    game: OpenGameData,
    device: torch.device | None = None,
    seq_len: int = 250,
    stride: int = 125,
    trans_penalty_ha: float = 3.0,
    trans_penalty_dead: float = 2.0,
    abstain_threshold: float = 0.45,
    enable_abstention: bool = True,
    apply_synthetic_dropout: bool = True,
    dropout_seed: int = VAL_DROPOUT_SEED,
) -> dict:
    """Evaluate a model on a full game with synthetic 41% ball blackout and Viterbi decoding.

    Args:
        model: PyTorch sequence model or fitted PossessionLGBM.
        game: OpenGameData container for the held-out match with precomputed features.
        device: Torch execution device (required for PyTorch models, ignored for LightGBM).
        seq_len: Slicing window length (PyTorch models only).
        stride: Slicing window step (PyTorch models only).
        trans_penalty_ha: Viterbi transition penalty between H and A.
        trans_penalty_dead: Viterbi transition penalty for Dead Ball D.
        abstain_threshold: Confidence threshold below which unobserved ball frames abstain.
        enable_abstention: Whether selective abstention ('X') is active.
        apply_synthetic_dropout: Apply realistic ~41% ball occlusion mask.
        dropout_seed: Deterministic seed for validation blackout generator.

    Returns:
        Evaluation dictionary from official evaluation/possession_metrics.py.
    """
    val_features, ball_vis = prepare_masked_game_features(
        game,
        target_ratio=VAL_DROPOUT_RATIO,
        seed=dropout_seed,
        apply_synthetic_dropout=apply_synthetic_dropout,
    )

    if isinstance(model, PossessionLGBM):
        emissions = model.predict_proba(val_features)  # probabilities (n_frames, 3)
        emission_type = "probs"
    else:
        if device is None:
            raise ValueError("A torch device is required to evaluate a PyTorch sequence model.")
        emissions = predict_sliding_window_logits(
            model=model,
            features=val_features,
            device=device,
            seq_len=seq_len,
            stride=stride,
        )
        emission_type = "logits"

    # Decode via Numba-accelerated Viterbi
    pred_codes = viterbi_decode(
        emissions=emissions,
        is_ball_visible=ball_vis,
        trans_penalty_ha=trans_penalty_ha,
        trans_penalty_dead=trans_penalty_dead,
        abstain_threshold=abstain_threshold,
        enable_abstention=enable_abstention,
        emission_type=emission_type,
    )

    # Score against official metrics
    return pm.evaluate(game.gt_labels, pred_codes)


def create_augmented_game_variants(
    game: OpenGameData,
    n_augmented: int = 2,
    seed: int = 42,
    dropout_ratio: float = 0.41,
    jitter_sigma: float = 0.25,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate original plus deterministic domain-augmented copies of a game sequence.

    Args:
        game: OpenGameData match container.
        n_augmented: Number of augmented variants to create (default: 2).
        seed: Random seed for deterministic reproducibility.
        dropout_ratio: Target ball occlusion dropout ratio (default: 0.41).
        jitter_sigma: Tracking coordinate noise standard deviation (default: 0.25 m).

    Returns:
        List of (features, labels) tuples containing the original game and augmented copies.
    """
    variants = [(game.features.copy(), game.gt_labels.copy())]

    for v_idx in range(1, n_augmented + 1):
        v_seed = seed + v_idx * 1000
        drop_mask = generate_ball_dropout_mask(game.n_frames, target_ratio=dropout_ratio, seed=v_seed)
        variants.append(
            apply_domain_augmentations(
                game.features.copy(),
                game.gt_labels.copy(),
                drop_mask,
                rng=np.random.RandomState(v_seed),
                jitter_sigma=jitter_sigma,
            )
        )

    return variants


def _print_split_header(
    title: str,
    train_games: list[OpenGameData],
    val_game: OpenGameData | None,
    train_note: str = "",
) -> None:
    """Print the banner describing one training split."""
    print(f"\n{'='*70}", flush=True)
    print(title, flush=True)
    print(f"Train games: {[g.game_id for g in train_games]}{train_note}", flush=True)
    if val_game is not None:
        print(f"Validation game: {val_game.game_id}", flush=True)
    print(f"{'='*70}", flush=True)


def train_lgbm_split(
    train_games: list[OpenGameData],
    val_game: OpenGameData | None,
    args: argparse.Namespace,
    fold_name: str,
) -> tuple[PossessionLGBM, dict | None]:
    """Train LightGBM on domain-augmented game variants and evaluate on held-out game."""
    _print_split_header(
        f"Training LightGBM split: {fold_name}",
        train_games,
        val_game,
        train_note=f" (with {N_AUGMENTED_VARIANTS} augmented variants each)",
    )

    train_feats_list = []
    train_labs_list = []
    for g in train_games:
        for feat_v, lab_v in create_augmented_game_variants(g, n_augmented=N_AUGMENTED_VARIANTS, seed=args.seed):
            train_feats_list.append(feat_v)
            train_labs_list.append(lab_v)

    n_total_frames = sum(len(f) for f in train_feats_list)
    print(f"Fitting LightGBM on {len(train_feats_list)} sequences ({n_total_frames} total frames)...", flush=True)

    n_threads = min(MAX_CPU_THREADS, os.cpu_count() or MAX_CPU_THREADS)
    model = PossessionLGBM(
        n_estimators=450,
        learning_rate=0.05,
        num_leaves=31,
        subsample_for_bin=50000,
        colsample_bytree=0.8,
        subsample=0.8,
        subsample_freq=1,
        min_child_samples=50,
        n_jobs=n_threads,
        random_state=args.seed,
        verbose=-1,
    )
    model.fit(train_feats_list, train_labs_list)
    print("LightGBM fitting completed.", flush=True)

    val_metrics = None
    if val_game is not None:
        print(f"Evaluating held-out game ({val_game.game_id}) under deterministic 41% synthetic ball dropout...", flush=True)
        val_metrics = evaluate_game(
            model=model,
            game=val_game,
            trans_penalty_ha=args.trans_penalty_ha,
            trans_penalty_dead=args.trans_penalty_dead,
            abstain_threshold=args.abstain_threshold,
        )
        score = val_metrics["spell_f1_mean"]
        macro_f1 = val_metrics["macro_f1"]
        acc = val_metrics["accuracy"]
        print(
            f"Validation [{val_game.game_id}] | Spell F1: {score:.4f} | "
            f"Macro F1: {macro_f1:.4f} | Acc: {acc:.4f}",
            flush=True,
        )

    return model, val_metrics


def train_single_split(
    train_games: list[OpenGameData],
    val_game: OpenGameData | None,
    args: argparse.Namespace,
    device: torch.device,
    fold_name: str,
) -> tuple[nn.Module, dict | None]:
    """Train model on a subset of games and evaluate on held-out game."""
    _print_split_header(f"Training split: {fold_name}", train_games, val_game)

    train_ds = OpenGameDataset(
        games=train_games,
        seq_len=args.seq_len,
        stride=args.stride,
        augment=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    if len(train_loader) == 0:
        raise ValueError(
            f"No full training batch available: {len(train_ds)} window(s) of {args.seq_len} frames "
            f"with batch size {args.batch_size}. Lower --batch-size or --seq-len."
        )

    class_weights = compute_class_weights([g.gt_labels for g in train_games], device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = build_model(
        arch=args.arch,
        input_dim=N_FEATURES,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_score = -1.0
    best_metrics = None
    best_weights = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"[{fold_name}] Ep {epoch:02d}/{args.epochs:02d}", leave=False)
        for x_b, y_b in pbar:
            x_b = x_b.to(device)
            y_b = y_b.to(device)

            optimizer.zero_grad()
            logits = model(x_b)  # (batch, seq_len, 3)

            ce_loss = criterion(logits.reshape(-1, N_CLASSES), y_b.reshape(-1))
            smooth_loss = temporal_smoothness_loss(logits)
            loss = ce_loss + args.smooth_weight * smooth_loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            loss_val = float(loss.item())
            total_loss += loss_val
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss_val:.4f}"})

        scheduler.step()
        avg_loss = total_loss / max(1, n_batches)

        # Validation on held-out game on cadence or final epoch
        should_eval = (val_game is not None) and (epoch % args.eval_every == 0 or epoch == args.epochs)
        if should_eval:
            metrics = evaluate_game(
                model=model,
                game=val_game,
                device=device,
                seq_len=args.seq_len,
                stride=args.stride,
                trans_penalty_ha=args.trans_penalty_ha,
                trans_penalty_dead=args.trans_penalty_dead,
                abstain_threshold=args.abstain_threshold,
            )
            score = metrics["spell_f1_mean"]
            macro_f1 = metrics["macro_f1"]
            acc = metrics["accuracy"]

            is_best = score > best_score
            if is_best:
                best_score = score
                best_metrics = metrics
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            marker = " [*BEST]" if is_best else ""
            print(
                f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {avg_loss:.4f} | "
                f"Val Spell F1: {score:.4f} | Macro F1: {macro_f1:.4f} | Acc: {acc:.4f}{marker}",
                flush=True,
            )
        else:
            print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {avg_loss:.4f}", flush=True)

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, best_metrics


def _save_torch_checkpoint(
    model: nn.Module,
    ckpt_file: Path,
    args: argparse.Namespace,
    **extra_metadata,
) -> None:
    """Save model weights together with the architecture metadata needed to rebuild the model."""
    torch.save({
        "model_state_dict": model.state_dict(),
        "arch": args.arch,
        "input_dim": N_FEATURES,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "num_classes": N_CLASSES,
        **extra_metadata,
    }, ckpt_file)


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject invalid CLI values up front with a readable message."""
    args.arch = args.arch.lower().strip()
    if args.arch not in ARCH_ALIASES:
        parser.error(f"--arch '{args.arch}' is not supported; choose one of {sorted(ARCH_ALIASES)}")
    try:
        validate_window_args(args.seq_len, args.stride, args.batch_size)
    except ValueError as error:
        parser.error(str(error))
    if args.epochs < 1 or args.eval_every < 1:
        parser.error("--epochs and --eval-every must be >= 1")
    if args.lr <= 0 or args.weight_decay < 0 or args.smooth_weight < 0:
        parser.error("--lr must be > 0; --weight-decay and --smooth-weight must be >= 0")
    if args.hidden_size < 1 or args.num_layers < 1 or not 0.0 <= args.dropout < 1.0:
        parser.error("--hidden-size and --num-layers must be >= 1 and --dropout must lie in [0, 1)")
    if args.trans_penalty_ha < 0 or args.trans_penalty_dead < 0:
        parser.error("--trans-penalty-ha and --trans-penalty-dead must be >= 0")
    if not 0.0 <= args.abstain_threshold <= 1.0:
        parser.error("--abstain-threshold must lie in [0, 1]")


def main() -> None:
    """Parse CLI arguments and run LOGO cross-validation or full training."""
    parser = argparse.ArgumentParser(description="Train and validate possession sequence models.")
    parser.add_argument("--mode", type=str, choices=["logo", "full"], default="logo",
                        help="'logo' for 3-fold Leave-One-Game-Out CV, 'full' for full training.")
    parser.add_argument("--arch", type=str, default="bi_gru",
                        help="Model architecture: 'bi_gru', 'tcn_1d' or 'lgbm' (default: bi_gru).")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs (default: 15).")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32).")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3).")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay (default: 1e-4).")
    parser.add_argument("--seq-len", type=int, default=250, help="Sequence length in frames (default: 250).")
    parser.add_argument("--stride", type=int, default=125, help="Slice stride in frames (default: 125).")
    parser.add_argument("--eval-every", type=int, default=3, help="Validation interval in epochs (default: 3).")
    parser.add_argument("--hidden-size", type=int, default=96, help="Hidden state size (default: 96).")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of GRU layers (default: 2).")
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout rate (default: 0.25).")
    parser.add_argument("--smooth-weight", type=float, default=0.1, help="Logit smoothness loss weight (default: 0.1).")
    parser.add_argument("--trans-penalty-ha", type=float, default=3.0, help="Viterbi H <-> A penalty (default: 3.0).")
    parser.add_argument("--trans-penalty-dead", type=float, default=2.0, help="Viterbi live <-> D penalty (default: 2.0).")
    parser.add_argument("--abstain-threshold", type=float, default=0.45, help="Abstention threshold (default: 0.45).")
    parser.add_argument("--data-dir", type=str, default=None, help="Root folder for open data.")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save checkpoints.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--device", type=str, default=None, help="'cuda', 'cpu', or None for auto.")
    args = parser.parse_args()
    _validate_args(parser, args)

    seed_everything(args.seed)
    try:
        device = resolve_device(args.device)
    except ValueError as error:
        parser.error(str(error))

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Ingest all 3 Metrica Open Games (cached to .npz for instant startup)
    print("Loading Metrica open games (Sample_Game_1, Sample_Game_2, Sample_Game_3)...", flush=True)
    games = [load_open_game_data(g_id, data_dir=args.data_dir) for g_id in [1, 2, 3]]
    for g in games:
        print(f"  {g.game_id}: {g.n_frames} frames ({g.n_frames / g.fps / 60.0:.1f} mins)", flush=True)

    is_lgbm = ARCH_ALIASES[args.arch] == "lgbm"

    if args.mode == "logo":
        fold_results: list[tuple[str, dict]] = []

        folds = [
            ("fold_1", [games[1], games[2]], games[0]),  # Train on 2 & 3, val on 1
            ("fold_2", [games[0], games[2]], games[1]),  # Train on 1 & 3, val on 2
            ("fold_3", [games[0], games[1]], games[2]),  # Train on 1 & 2, val on 3
        ]

        for fname, train_g, val_g in folds:
            if is_lgbm:
                model, metrics = train_lgbm_split(
                    train_games=train_g,
                    val_game=val_g,
                    args=args,
                    fold_name=fname,
                )
                ckpt_file = output_path / f"lgbm_{fname}.joblib"
                model.save(ckpt_file)
                print(f"Saved checkpoint: {ckpt_file}", flush=True)
            else:
                model, metrics = train_single_split(
                    train_games=train_g,
                    val_game=val_g,
                    args=args,
                    device=device,
                    fold_name=fname,
                )
                # Save fold checkpoint
                ckpt_file = output_path / f"{args.arch}_{fname}.pt"
                _save_torch_checkpoint(model, ckpt_file, args, val_game=val_g.game_id, val_metrics=metrics)
                print(f"Saved checkpoint: {ckpt_file}", flush=True)

            if metrics:
                fold_results.append((val_g.game_id, metrics))

        print(f"\n{'='*70}", flush=True)
        summary_title = "LEAVE-ONE-GAME-OUT (LOGO) CV RESULTS SUMMARY"
        if is_lgbm:
            summary_title += " (LightGBM)"
        print(summary_title, flush=True)
        print(f"{'='*70}", flush=True)
        if not fold_results:
            print("No fold produced validation metrics.", flush=True)
        else:
            fold_scores = [m["spell_f1_mean"] for _, m in fold_results]
            macro_scores = [m["macro_f1"] for _, m in fold_results]
            for i, (game_id, _) in enumerate(fold_results):
                print(
                    f"Fold {i+1} (Val: {game_id}): Spell F1 = {fold_scores[i]:.4f} | "
                    f"Macro F1 = {macro_scores[i]:.4f}",
                    flush=True,
                )
            print(f"{'-'*70}", flush=True)
            print(f"Mean CV Spell F1: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})", flush=True)
            print(f"Mean CV Macro F1: {np.mean(macro_scores):.4f} (+/- {np.std(macro_scores):.4f})", flush=True)
        print(f"{'='*70}", flush=True)

    elif args.mode == "full":
        if is_lgbm:
            model, _ = train_lgbm_split(
                train_games=games,
                val_game=None,
                args=args,
                fold_name="full_3_games",
            )
            ckpt_file = output_path / "lgbm_full.joblib"
            model.save(ckpt_file)
            print(f"\nSaved full model checkpoint: {ckpt_file}", flush=True)
        else:
            model, _ = train_single_split(
                train_games=games,
                val_game=None,
                args=args,
                device=device,
                fold_name="full_3_games",
            )
            ckpt_file = output_path / f"{args.arch}_full.pt"
            _save_torch_checkpoint(model, ckpt_file, args)
            print(f"\nSaved full model checkpoint: {ckpt_file}", flush=True)


if __name__ == "__main__":
    main()
