"""Shared runtime helpers for training, ensemble tuning and match inference.

Single home for the logic the three CLIs have in common: deterministic seeding, device
selection, checkpoint loading and overlapping sliding-window prediction.
"""

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.ml_pipeline.features import N_FEATURES
from src.ml_pipeline.models import PossessionLGBM, build_model
from src.ml_pipeline.viterbi import softmax

N_CLASSES = 3
MAX_CPU_THREADS = 4

# Architecture hyper-parameters assumed when a checkpoint does not record them
_DEFAULT_HIDDEN_SIZE = 96
_DEFAULT_NUM_LAYERS = 2
_DEFAULT_DROPOUT = 0.25


def seed_everything(seed: int = 42) -> None:
    """Seed every random number generator and request deterministic cuDNN kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_arg: str | None = None) -> torch.device:
    """Resolve the execution device ('cuda', 'cpu', or auto-detect) and cap CPU threads.

    Raises:
        ValueError: If the device string is invalid or requests CUDA when it is unavailable.
    """
    if device_arg:
        try:
            device = torch.device(device_arg)
        except RuntimeError as error:
            raise ValueError(f"Invalid --device '{device_arg}': {error}") from error
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError(f"--device '{device_arg}' requested but CUDA is not available.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    if device.type == "cpu":
        n_threads = min(MAX_CPU_THREADS, os.cpu_count() or MAX_CPU_THREADS)
        torch.set_num_threads(n_threads)
        print(f"Configured PyTorch CPU threads: {n_threads}", flush=True)
    return device


def validate_window_args(seq_len: int, stride: int, batch_size: int) -> None:
    """Check sliding-window settings; a stride above seq_len would leave frames unpredicted.

    Raises:
        ValueError: If any value is out of range.
    """
    if seq_len < 1:
        raise ValueError(f"Sequence length must be >= 1, got {seq_len}")
    if not 1 <= stride <= seq_len:
        raise ValueError(f"Stride must satisfy 1 <= stride <= sequence length ({seq_len}), got {stride}")
    if batch_size < 1:
        raise ValueError(f"Batch size must be >= 1, got {batch_size}")


def sliding_window_spans(n_frames: int, seq_len: int, stride: int) -> list[tuple[int, int]]:
    """Half-open [start, end) windows that cover every frame of a sequence at least once.

    The last window is anchored to the end of the sequence. A sequence shorter than seq_len
    is covered by one window spanning all of it.
    """
    if n_frames <= 0:
        return []
    if n_frames <= seq_len:
        return [(0, n_frames)]

    max_start = n_frames - seq_len
    starts = list(range(0, max_start + 1, stride))
    if starts[-1] < max_start:
        starts.append(max_start)
    return [(start, start + seq_len) for start in starts]


def predict_sliding_window_logits(
    model: nn.Module,
    features: np.ndarray,
    device: torch.device,
    seq_len: int = 250,
    stride: int = 125,
    batch_size: int = 64,
) -> np.ndarray:
    """Run a sequence model across a full sequence, averaging logits over overlapping windows.

    Args:
        model: Trained PyTorch sequence model.
        features: Frame features array of shape (n_frames, n_features).
        device: Torch execution device.
        seq_len: Slicing sequence length in frames.
        stride: Slicing step in frames (1 <= stride <= seq_len).
        batch_size: Window batch size.

    Returns:
        Averaged logits array of shape (n_frames, 3), dtype float32.
    """
    validate_window_args(seq_len, stride, batch_size)
    model.eval()

    features = np.ascontiguousarray(features, dtype=np.float32)
    n_frames = len(features)
    accum_logits = np.zeros((n_frames, N_CLASSES), dtype=np.float32)
    counts = np.zeros(n_frames, dtype=np.float32)
    spans = sliding_window_spans(n_frames, seq_len, stride)

    with torch.no_grad():
        for b_start in range(0, len(spans), batch_size):
            batch_spans = spans[b_start : b_start + batch_size]
            batch_data = np.stack([features[s:e] for s, e in batch_spans], axis=0)
            batch_logits = model(torch.from_numpy(batch_data).to(device)).cpu().numpy()

            for i, (s, e) in enumerate(batch_spans):
                accum_logits[s:e] += batch_logits[i]
                counts[s:e] += 1.0

    return accum_logits / np.maximum(counts, 1.0)[:, None]


def predict_sliding_window_probabilities(
    model: nn.Module,
    features: np.ndarray,
    device: torch.device,
    seq_len: int = 250,
    stride: int = 125,
    batch_size: int = 64,
) -> np.ndarray:
    """Per-frame softmax probabilities, shape (n_frames, 3), float32. See predict_sliding_window_logits."""
    avg_logits = predict_sliding_window_logits(
        model=model,
        features=features,
        device=device,
        seq_len=seq_len,
        stride=stride,
        batch_size=batch_size,
    )
    return softmax(avg_logits).astype(np.float32)


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
    default_arch: str | None = None,
) -> nn.Module | PossessionLGBM:
    """Load a model from a saved checkpoint file (.pt or .joblib).

    Supports:
    - PyTorch sequence models (PossessionBiGRU, PossessionTCN1D) saved as .pt, either as a
      training checkpoint dict (with 'model_state_dict' and architecture metadata) or as a
      bare state dict.
    - LightGBM tabular models (PossessionLGBM) saved as .joblib.

    Args:
        checkpoint_path: Path to the checkpoint file.
        device: Torch execution device.
        default_arch: Architecture to assume when the checkpoint does not record one. When
            omitted, it is inferred from the file path ('tcn' -> tcn_1d, otherwise bi_gru).

    Returns:
        The model in eval mode on device (nn.Module) or a PossessionLGBM instance.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        ValueError: If the file content is not a recognized checkpoint.
    """
    p = Path(checkpoint_path)
    if not p.is_file():
        raise FileNotFoundError(
            f"Checkpoint file not found: {p}\n"
            "Train it first with: python src/ml_pipeline/train_logo.py --mode {logo,full} --arch <arch>"
        )
    if p.suffix == ".joblib" or "lgbm" in p.name.lower():
        return PossessionLGBM.load(p)

    ckpt = torch.load(p, map_location=device)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unrecognized checkpoint format in {p}: expected a dict, got {type(ckpt).__name__}")

    if "model_state_dict" in ckpt:
        meta, state_dict = ckpt, ckpt["model_state_dict"]
    else:
        meta, state_dict = {}, ckpt  # bare state dict

    arch = meta.get("arch") or default_arch or ("tcn_1d" if "tcn" in str(p).lower() else "bi_gru")
    model = build_model(
        arch=arch,
        input_dim=meta.get("input_dim", N_FEATURES),
        hidden_size=meta.get("hidden_size", _DEFAULT_HIDDEN_SIZE),
        num_layers=meta.get("num_layers", _DEFAULT_NUM_LAYERS),
        dropout=meta.get("dropout", _DEFAULT_DROPOUT),
        num_classes=meta.get("num_classes", N_CLASSES),
    )
    if not isinstance(model, nn.Module):
        raise ValueError(f"Checkpoint {p} declares architecture '{arch}', which is not a PyTorch model.")

    model = model.to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model
