"""Inference CLI for competition match (BAR-ATH).

Loads match assets via games.py, src/data_loader.py, src/ball_tracking.py,
and starter/load_smart_tagging.py, extracts canonical 14-dimensional features,
runs model inference (single checkpoint or LOGO 3-fold probability ensemble),
decodes each half independently via Viterbi decoding, validates format,
and writes submission to results/lamasia_helvetia_bigru.csv.
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
from src.data_loader import load_raw_tracking_data
from src.dead_ball import STOPPAGE_CODES, STOPPAGE_SET_PIECE_TYPES
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
    """Load model architecture and weights from saved checkpoint file."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    arch = ckpt.get("arch", "bi_gru")
    hidden_size = ckpt.get("hidden_size", 96)
    num_layers = ckpt.get("num_layers", 2)
    dropout = ckpt.get("dropout", 0.25)
    input_dim = ckpt.get("input_dim", N_FEATURES)

    model = build_model(
        arch=arch,
        input_dim=input_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def predict_half_probabilities(
    models: list[nn.Module],
    features: np.ndarray,
    device: torch.device,
    seq_len: int = 250,
    stride: int = 125,
    batch_size: int = 64,
) -> np.ndarray:
    """Predict probabilities across a half, averaging over models if ensemble."""
    n_frames = len(features)
    half_probs = np.zeros((n_frames, 3), dtype=np.float32)

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

    for model in models:
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

        # Compute softmax probabilities
        shift = avg_logits - np.max(avg_logits, axis=-1, keepdims=True)
        exp_vals = np.exp(shift)
        m_probs = exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)

        half_probs += m_probs / float(len(models))

    return half_probs


def main():
    parser = argparse.ArgumentParser(description="Run possession inference on the competition match.")
    parser.add_argument("--checkpoint", type=str, default="models/bi_gru_full.pt",
                        help="Path to single model checkpoint (used if --ensemble is False).")
    parser.add_argument("--ensemble", action="store_true",
                        help="Ensemble across the 3 LOGO fold checkpoints (bi_gru_fold_1/2/3.pt).")
    parser.add_argument("--models-dir", type=str, default="models",
                        help="Directory containing fold checkpoints when --ensemble is set.")
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

    # 5. Load model(s)
    models = []
    if args.ensemble:
        models_dir = Path(args.models_dir)
        fold_files = [
            models_dir / "bi_gru_fold_1.pt",
            models_dir / "bi_gru_fold_2.pt",
            models_dir / "bi_gru_fold_3.pt",
        ]
        print(f"Loading 3-fold LOGO ensemble from: {[str(f) for f in fold_files]}")
        for f in fold_files:
            if not f.is_file():
                raise FileNotFoundError(f"Missing ensemble checkpoint: {f}")
            models.append(load_model_from_checkpoint(f, device))
    else:
        ckpt_path = Path(args.checkpoint)
        print(f"Loading single checkpoint from: {ckpt_path}")
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
        models.append(load_model_from_checkpoint(ckpt_path, device))

    # 6. Independent half-by-half decoding
    half_blocks = half_video_blocks(GAME)
    final_codes = np.full(n_expected, 2, dtype=np.int8)  # default Dead Ball (D = 2)

    for half_num, h_start, h_end in half_blocks:
        print(f"\nProcessing Half {half_num} (frames {h_start}..{h_end}, span: {h_end - h_start + 1})...")
        h_features = features[h_start : h_end + 1]
        h_ball_vis = ~ball_res.is_ball_uncertain[h_start : h_end + 1]

        # Predict probabilities
        h_probs = predict_half_probabilities(
            models=models,
            features=h_features,
            device=device,
            seq_len=args.seq_len,
            stride=args.stride,
            batch_size=args.batch_size,
        )

        # Decode via Viterbi
        decoded_half = viterbi_decode(
            emissions=h_probs,
            is_ball_visible=h_ball_vis,
            trans_penalty_ha=args.trans_penalty_ha,
            trans_penalty_dead=args.trans_penalty_dead,
            abstain_threshold=args.abstain_threshold,
            enable_abstention=not args.no_abstain,
        )

        final_codes[h_start : h_end + 1] = decoded_half

    # 7. Convert codes to string labels and validate submission contract
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

    # 8. Football analytics summary
    fps = GAMES[GAME]["fps"]
    n_spells, mean_dur = describe(final_codes, fps)
    print(f"\nSubmission claims {n_spells} possessions with mean duration {mean_dur:.1f} s")
    print("Label distribution:")
    for idx, name in enumerate(pm.LABELS):
        count = int(np.sum(final_codes == idx))
        print(f"  {name}: {count:>7} frames ({100.0 * count / n_expected:5.1f}%)")

    # 9. Write submission CSV
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_submission.to_csv(out_path, index=False)
    print(f"\nSuccessfully saved submission to: {out_path}")


if __name__ == "__main__":
    main()
