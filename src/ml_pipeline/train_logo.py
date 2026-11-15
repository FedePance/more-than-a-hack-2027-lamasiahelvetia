"""Leave-One-Game-Out (LOGO) Cross-Validation and Full Training CLI.

Supports:
1. --mode logo: 3-fold Leave-One-Game-Out cross validation across Metrica Sample Games 1-3.
   Trains on 2 games with fast on-the-fly augmentations and evaluates on the held-out game
   under deterministic 41% synthetic ball dropout using Numba-accelerated Viterbi decoding
   and official metrics (spell_f1_mean, macro_f1). Saves checkpoints to models/bi_gru_fold_{k}.pt.
2. --mode full: Trains on all 3 games combined and saves to models/bi_gru_full.pt.
"""

import argparse
import os
import random
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
    generate_ball_dropout_mask,
    load_open_game_data,
)
from src.ml_pipeline.features import N_FEATURES
from src.ml_pipeline.models import build_model
from src.ml_pipeline.viterbi import viterbi_decode


def seed_everything(seed: int = 42) -> None:
    """Set random seed across all libraries for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(labels_list: list[np.ndarray], device: torch.device) -> torch.Tensor:
    """Compute balanced inverse class frequency weights for CrossEntropyLoss."""
    all_labels = np.concatenate(labels_list)
    counts = np.bincount(all_labels, minlength=3).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    total = float(len(all_labels))
    weights = total / (3.0 * counts)
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


def predict_full_game_logits(
    model: nn.Module,
    features: np.ndarray,
    device: torch.device,
    seq_len: int = 250,
    stride: int = 125,
    batch_size: int = 64,
) -> np.ndarray:
    """Run model inference across full match frames using overlapping window averaging.

    Args:
        model: Trained PyTorch sequence model.
        features: Frame features array of shape (n_frames, n_features).
        device: Torch execution device.
        seq_len: Slicing sequence length in frames.
        stride: Slicing step in frames.
        batch_size: Window batch size.

    Returns:
        Averaged logits array of shape (n_frames, 3).
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
    return accum_logits / counts


def evaluate_game(
    model: nn.Module,
    game: OpenGameData,
    device: torch.device,
    seq_len: int = 250,
    stride: int = 125,
    trans_penalty_ha: float = 3.0,
    trans_penalty_dead: float = 2.0,
    abstain_threshold: float = 0.45,
    enable_abstention: bool = True,
    apply_synthetic_dropout: bool = True,
    dropout_seed: int = 42,
) -> dict:
    """Evaluate model on a full game with synthetic 41% ball blackout and Viterbi decoding.

    Args:
        model: PyTorch model in eval mode.
        game: OpenGameData container for the held-out match with precomputed features.
        device: Torch execution device.
        seq_len: Slicing window length.
        stride: Slicing window step.
        trans_penalty_ha: Viterbi transition penalty between H and A.
        trans_penalty_dead: Viterbi transition penalty for Dead Ball D.
        abstain_threshold: Confidence threshold below which unobserved ball frames abstain.
        enable_abstention: Whether selective abstention ('X') is active.
        apply_synthetic_dropout: Apply realistic ~41% ball occlusion mask.
        dropout_seed: Deterministic seed for validation blackout generator.

    Returns:
        Evaluation dictionary from official evaluation/possession_metrics.py.
    """
    n_frames = game.n_frames
    ball_vis = game.is_ball_visible.copy()

    if apply_synthetic_dropout:
        val_dropout_mask = generate_ball_dropout_mask(n_frames, target_ratio=0.41, seed=dropout_seed)
        ball_vis = ball_vis & ~val_dropout_mask

    # Fast feature masking from precomputed base features (< 2 ms)
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

    # Predict full game logits
    logits = predict_full_game_logits(
        model=model,
        features=val_features,
        device=device,
        seq_len=seq_len,
        stride=stride,
    )

    # Decode via Numba-accelerated Viterbi
    pred_codes = viterbi_decode(
        emissions=logits,
        is_ball_visible=ball_vis,
        trans_penalty_ha=trans_penalty_ha,
        trans_penalty_dead=trans_penalty_dead,
        abstain_threshold=abstain_threshold,
        enable_abstention=enable_abstention,
    )

    # Score against official metrics
    metrics = pm.evaluate(game.gt_labels, pred_codes)
    return metrics


def train_single_split(
    train_games: list[OpenGameData],
    val_game: OpenGameData | None,
    args: argparse.Namespace,
    device: torch.device,
    fold_name: str,
) -> tuple[nn.Module, dict | None]:
    """Train model on a subset of games and evaluate on held-out game."""
    print(f"\n{'='*70}", flush=True)
    print(f"Training split: {fold_name}", flush=True)
    print(f"Train games: {[g.game_id for g in train_games]}", flush=True)
    if val_game:
        print(f"Validation game: {val_game.game_id}", flush=True)
    print(f"{'='*70}", flush=True)

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

            ce_loss = criterion(logits.view(-1, 3), y_b.view(-1))
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


def main():
    parser = argparse.ArgumentParser(description="Train and validate possession sequence models.")
    parser.add_argument("--mode", type=str, choices=["logo", "full"], default="logo",
                        help="'logo' for 3-fold Leave-One-Game-Out CV, 'full' for full training.")
    parser.add_argument("--arch", type=str, default="bi_gru", help="Model architecture (default: bi_gru).")
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

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Ingest all 3 Metrica Open Games (cached to .npz for instant startup)
    print("Loading Metrica open games (Sample_Game_1, Sample_Game_2, Sample_Game_3)...", flush=True)
    games = [load_open_game_data(g_id, data_dir=args.data_dir) for g_id in [1, 2, 3]]
    for g in games:
        print(f"  {g.game_id}: {g.n_frames} frames ({g.n_frames / g.fps / 60.0:.1f} mins)", flush=True)

    if args.mode == "logo":
        fold_scores = []
        macro_scores = []

        folds = [
            ("fold_1", [games[1], games[2]], games[0]),  # Train on 2 & 3, val on 1
            ("fold_2", [games[0], games[2]], games[1]),  # Train on 1 & 3, val on 2
            ("fold_3", [games[0], games[1]], games[2]),  # Train on 1 & 2, val on 3
        ]

        for fname, train_g, val_g in folds:
            model, metrics = train_single_split(
                train_games=train_g,
                val_game=val_g,
                args=args,
                device=device,
                fold_name=fname,
            )

            # Save fold checkpoint
            ckpt_file = output_path / f"bi_gru_{fname}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "arch": args.arch,
                "input_dim": N_FEATURES,
                "hidden_size": args.hidden_size,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "val_game": val_g.game_id,
                "val_metrics": metrics,
            }, ckpt_file)
            print(f"Saved checkpoint: {ckpt_file}", flush=True)

            if metrics:
                fold_scores.append(metrics["spell_f1_mean"])
                macro_scores.append(metrics["macro_f1"])

        print(f"\n{'='*70}", flush=True)
        print("LEAVE-ONE-GAME-OUT (LOGO) CV RESULTS SUMMARY", flush=True)
        print(f"{'='*70}", flush=True)
        for i, (fname, _, val_g) in enumerate(folds):
            print(f"Fold {i+1} (Val: {val_g.game_id}): Spell F1 = {fold_scores[i]:.4f} | Macro F1 = {macro_scores[i]:.4f}", flush=True)
        print(f"{'-'*70}", flush=True)
        print(f"Mean CV Spell F1: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})", flush=True)
        print(f"Mean CV Macro F1: {np.mean(macro_scores):.4f} (+/- {np.std(macro_scores):.4f})", flush=True)
        print(f"{'='*70}", flush=True)

    elif args.mode == "full":
        model, _ = train_single_split(
            train_games=games,
            val_game=None,
            args=args,
            device=device,
            fold_name="full_3_games",
        )
        ckpt_file = output_path / "bi_gru_full.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "arch": args.arch,
            "input_dim": N_FEATURES,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
        }, ckpt_file)
        print(f"\nSaved full model checkpoint: {ckpt_file}", flush=True)


if __name__ == "__main__":
    main()
