# More Than A Hack 2027 - Phase 1: Ball Possession Identification

**Team:** La MasIA Helvetia  
**Challenge:** More Than A Hack 2027 (Phase 1 - Automatic Ball Possession Identification)  
**Match Reference:** FC Barcelona vs Athletic Club (`BAR-ATH`)  
**Target Video Horizon:** 199,319 frames at 25 fps (~132.9 minutes)  
**Alphabet:** Home Possession (`H`), Away Possession (`A`), Dead Ball / Stoppage (`D`), Selective Abstention (`X`)

---

## 1. Overview & Competition Compliance

This repository implements the end-to-end ball possession identification system developed by team **La MasIA Helvetia** for Phase 1 of the *More Than A Hack 2027* sports analytics competition. The goal is to accurately segment continuous match video into team possession spells (`H`, `A`) and dead-ball stoppages (`D`) at a native temporal resolution of 25 frames per second, while strategically emitting selective abstention (`X`) during unresolvable tracking blackouts.

### Strict Zero-Leakage Governance
The system strictly enforces zero-leakage competition data isolation:
- **Zero Training on Match Data:** Neither the competition match tracking data (`BAR-ATH`) nor any public ground-truth validation slice was used to train or adjust model parameters.
- **External Open-Data Training Only:** All supervised sequence and tabular models are trained exclusively on external, open-access football tracking datasets (Metrica Sample Games 1, 2, and 3 via `kloppy`), rigorously evaluated via 3-fold Leave-One-Game-Out (LOGO) cross-validation.
- **Robust Domain Invariance:** The feature extraction pipeline projects all match coordinates and kinematics into a 14-dimensional, domain-invariant coordinate system that is identical between open training matches and the competition match.
- **Purely Physical Post-Processing:** All rule-based modules and hysteresis thresholds are derived strictly from football physics (player reach, athletic running speeds, pitch geometry).

---

## 2. Repository Structure

```
more-than-a-hack-2027-lamasiahelvetia/
├── .devcontainer/                  # Dev Container & Docker environment configuration
│   ├── Dockerfile                  # Container environment with PyTorch and dependencies
│   ├── devcontainer.json           # VS Code configuration with bind-mount points
│   └── post-create.sh              # Container initialization script
├── data/                           # Match data mount root (tactical & broadcast; git-ignored)
│   ├── tactical/                   # Authority source: BAR-ATH_tactical (.mp4, _atd.txt, _atd.xml, _smart_tagging.xml)
│   └── broadcast/                  # Optional broadcast camera source (BAR-ATH_broadcast)
├── docs/                           # Documentation and technical reports
│   └── technical_report.md         # Comprehensive system architecture & methodology report
├── evaluation/                     # Official evaluation suite & submission checkers
│   ├── possession_metrics.py       # Official scorer implementation (spellF1, macroF1, transition metrics)
│   ├── run_evaluation.py           # Benchmark scoreboard generation against ground truth
│   └── validate_submission.py      # Pre-submission format, shape, and integrity validator
├── external_open/                  # External open tracking data directory (git-ignored)
│   ├── data/                       # Root for Metrica Sample Games (Sample_Game_1, 2, 3)
│   └── .cache/                     # Fast-loading precomputed .npz archives (< 0.2 s startup)
├── models/                         # Pre-trained production checkpoints (bi_gru_full.pt, tcn_1d_full.pt)
├── results/                        # Generated submission CSV files (frame, label)
├── src/                            # Deterministic physics and kinematic pipeline
│   ├── __init__.py                 # Core package exports
│   ├── ball_tracking.py            # PCHIP trajectory reconstruction & ~41% blackout flagging
│   ├── config.py                   # Centralized PipelineConfig dataclass with physical parameters
│   ├── data_loader.py              # Streaming raw ATD tracking and metadata parser
│   ├── dead_ball.py                # Smart Tagging dilation, boundary latching, and p80 kinematics
│   ├── mask_utils.py               # Shared run-length helpers for boolean frame masks
│   ├── possession_engine.py        # Hysteresis state machine, pass preservation, and spell regularizer
│   ├── run_pipeline.py             # CLI entrypoint for the deterministic physics pipeline
│   └── ml_pipeline/                # Supervised sequence and tabular learning framework
│       ├── __init__.py             # ML pipeline exports
│       ├── dataset_open.py         # Open data loader, GT derivation, and on-the-fly augmentations
│       ├── features.py             # Canonical 14-dimensional domain-invariant feature extractor
│       ├── infer_match.py          # Match inference CLI (ensembles, hybrid dead-ball, Viterbi)
│       ├── inference.py            # Shared seeding, device, checkpoint loading, sliding-window prediction
│       ├── train_logo.py           # Leave-One-Game-Out CV & full training CLI
│       ├── tune_ensemble_logo.py   # LOGO tuning of BiGRU / TCN-1D ensemble blending weights
│       ├── viterbi.py              # Numba-accelerated 3-state Viterbi decoder (float64) with selective X
│       └── models/                 # Model architectures registry
│           ├── __init__.py         # Model factory (build_model)
│           ├── bi_gru.py           # Bidirectional GRU with residual MLP classification head
│           ├── lgbm_model.py       # Multi-scale rolling LightGBM with scipy.ndimage features
│           └── tcn_1d.py           # Non-causal symmetric 1D Dilated Temporal Convolutional Network
├── starter/                        # Reference starter data loaders
│   ├── load_atd.py                 # FIFA/EPTS raw tracking data loader
│   └── load_smart_tagging.py       # SportsCode XML Smart Tagging parser
├── .gitignore                      # Git exclusion rules for private data, models, and artifacts
├── games.py                        # Central match timeline and asset path resolution constants
└── requirements.txt                # Pinned production dependencies (PyTorch CPU, LightGBM, Numba)
```

---

## 3. Environment & Data Setup

### Option A: VS Code Dev Container (Recommended)
The project includes a ready-to-use Dev Container pre-configured with Python 3.11, PyTorch (CPU wheel), Numba, LightGBM, and scientific computing packages:
1. Open the repository in **VS Code**.
2. When prompted, select **Reopen in Container** (or press `F1` and choose `Dev Containers: Reopen in Container`).
3. Private match data can be mounted automatically by setting the host path in `.devcontainer/devcontainer.json`:
   ```json
   "mounts": [
     "source=/path/to/host/mtah-data,target=/mtah_private_data,type=bind"
   ]
   ```
4. The container automatically maps `MTAH_DATA_DIR=/mtah_private_data` and activates `/opt/venv`.

### Option B: Local Python Virtual Environment
To run on a local workstation:
```bash
# 1. Create and activate a clean virtual environment
python3 -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate

# 2. Upgrade core packaging tools
pip install --upgrade pip setuptools wheel

# 3. Install pinned dependencies
pip install -r requirements.txt
```

### Match Data Placement (`BAR-ATH`)
The match data bundle contains high-resolution video and EPTS tracking. It can be placed in either of two ways:
1. **Direct Workspace Directory:** Unpack into `./data/` matching the canonical structure:
   ```
   data/
   └── tactical/
       ├── BAR-ATH_tactical.mp4
       ├── BAR-ATH_tactical_atd.xml
       ├── BAR-ATH_tactical_atd.txt
       ├── BAR-ATH_tactical_homography.csv
       └── BAR-ATH_tactical_smart_tagging.xml
   ```
2. **Environment Variable:** Point `MTAH_DATA_DIR` to any external directory containing `tactical/`:
   ```bash
   export MTAH_DATA_DIR="/external/storage/mtah2027"       # Linux / macOS
   $env:MTAH_DATA_DIR = "D:\storage\mtah2027"             # Windows PowerShell
   ```

### External Open-Data Setup (Metrica Sample Games 1–3)
To perform Leave-One-Game-Out cross-validation or train supervised ML models from scratch, external open tracking data is required:
1. Clone or download the Metrica Sports public sample data into `external_open/data/`:
   ```
   external_open/data/
   ├── Sample_Game_1/
   │   ├── Sample_Game_1_RawTrackingData_Home_Team.csv
   │   ├── Sample_Game_1_RawTrackingData_Away_Team.csv
   │   └── Sample_Game_1_RawEventsData.csv
   ├── Sample_Game_2/
   │   ├── Sample_Game_2_RawTrackingData_Home_Team.csv
   │   ├── Sample_Game_2_RawTrackingData_Away_Team.csv
   │   └── Sample_Game_2_RawEventsData.csv
   └── Sample_Game_3/
       ├── Sample_Game_3_tracking.txt
       ├── Sample_Game_3_metadata.xml
       └── Sample_Game_3_events.json
   ```
2. On first execution, `src/ml_pipeline/dataset_open.py` automatically parses tracking files, synchronizes event ground truth, precomputes the 14 base features, and writes compressed archives to `external_open/.cache/*.npz` for instantaneous subsequent loads.

---

## 4. Usage & Reproducible Workflows

### 4.0 Reproducing the Final Submission (`results/lamasia_helvetia.csv`)
With the competition match assets placed under `./data/tactical/` (or `MTAH_DATA_DIR`), regenerate the exact submission file using the shipped pre-trained checkpoints (`models/bi_gru_full.pt` and `models/tcn_1d_full.pt`, blended 50/50 by default) and verify its format:

```bash
python src/ml_pipeline/infer_match.py \
  --checkpoints models/bi_gru_full.pt models/tcn_1d_full.pt \
  --hybrid-dead-ball \
  --output results/lamasia_helvetia.csv

python evaluation/validate_submission.py results/lamasia_helvetia.csv
```

### 4.1 Running the Deterministic Physics/Heuristic Pipeline
The deterministic physics pipeline reconstructs ball trajectories, extracts player kinematics, identifies dead-ball states via Smart Tagging and player speed percentiles, and resolves possession via a hysteresis state machine:

```bash
python src/run_pipeline.py \
  --data-dir ./data \
  --output results/lamasia_helvetia_physics.csv
```

**Key Execution Stages:**
- **Streaming ATD Loader:** Streams raw FIFA/EPTS text tracking directly into NumPy memory.
- **Ball Trajectory Reconstruction (`src/ball_tracking.py`):** Filters teleportation spikes, applies Savitzky-Golay filtering, and reconstructs short occlusions (< 35 frames) with PCHIP interpolation and analytic derivatives.
- **Dead-Ball Identification (`src/dead_ball.py`):** Fuses Smart Tagging event intervals, pitch boundary out-of-bounds latching, and collective 80th-percentile player speed drops.
- **Possession State Machine (`src/possession_engine.py`):** Enforces pass-preserving evidence accumulation, resolves kickoffs dynamically, abstains (`X`) on contested scrambles, and regularizes micro-flips (< 35 frames).

`--source` accepts only `tactical`: submissions are indexed to the tactical video, and the half boundaries and Smart Tagging exist only on that timeline.

---

### 4.2 Machine Learning Model Training (LOGO CV & Full Training)
The ML pipeline supports Leave-One-Game-Out cross-validation across Metrica Sample Games 1–3, as well as full open-data training across all three games.

#### Supported Model Architectures:
- `bi_gru`: 2-layer Bidirectional GRU with residual MLP classification head (`hidden_size=96`).
- `tcn_1d`: Symmetric non-causal 1D Dilated Temporal Convolutional Network with exponential dilations (1, 2, 4, 8, 16) and GroupNorm.
- `lgbm`: Symmetric rolling multi-scale temporal LightGBM classifier with 218 vectorized features.

#### Running Leave-One-Game-Out (LOGO) Cross-Validation:
```bash
# 3-Fold LOGO Cross-Validation on Bidirectional GRU
python src/ml_pipeline/train_logo.py \
  --mode logo \
  --arch bi_gru \
  --epochs 15 \
  --batch-size 32 \
  --output-dir models

# 3-Fold LOGO Cross-Validation on 1D Dilated TCN
python src/ml_pipeline/train_logo.py \
  --mode logo \
  --arch tcn_1d \
  --epochs 15 \
  --batch-size 32 \
  --output-dir models

# 3-Fold LOGO Cross-Validation on Multi-Scale Rolling LightGBM
python src/ml_pipeline/train_logo.py \
  --mode logo \
  --arch lgbm \
  --output-dir models
```

#### Training Full Production Checkpoints on All 3 Games:
```bash
# Train full BiGRU model on Games 1, 2, and 3 combined
python src/ml_pipeline/train_logo.py \
  --mode full \
  --arch bi_gru \
  --epochs 15 \
  --output-dir models

# Train full TCN model
python src/ml_pipeline/train_logo.py \
  --mode full \
  --arch tcn_1d \
  --epochs 15 \
  --output-dir models

# Train full LightGBM model
python src/ml_pipeline/train_logo.py \
  --mode full \
  --arch lgbm \
  --output-dir models
```

Checkpoints are saved to `models/{arch}_fold_{1,2,3}.pt` and `models/{arch}_full.pt` (or `.joblib` for LightGBM).

#### Tuning Ensemble Blending Weights (BiGRU + TCN-1D):
Once the fold checkpoints of both architectures exist, the blending weights are chosen on the held-out open games only (never on the competition match):
```bash
python src/ml_pipeline/tune_ensemble_logo.py --models-dir models
```
The script scores a fixed grid of weight pairs with the official metrics under the 41% synthetic ball blackout and prints a ready-to-run `infer_match.py` command for the best pair. It stops immediately, before loading any data, if a fold checkpoint is missing.

---

### 4.3 Running Match Inference (`src/ml_pipeline/infer_match.py`)
Match inference executes feature extraction on `BAR-ATH`, sliding-window neural inference, optional kinematic dead-ball fusion, and independent half-by-half Viterbi decoding.

#### Primary Hybrid Production Inference (50/50 BiGRU + 1D-TCN):
```bash
python src/ml_pipeline/infer_match.py \
  --checkpoints models/bi_gru_full.pt models/tcn_1d_full.pt \
  --hybrid-dead-ball \
  --output results/lamasia_helvetia.csv
```
When multiple checkpoints are passed without `--weights`, `infer_match.py` defaults to uniform weighting (equal 50/50 blend for two models).

#### Optional CLI Execution Modes & Flags:
- `--checkpoint <path>`: Run inference with a single checkpoint (e.g. `models/bi_gru_full.pt`).
- `--ensemble --arch <arch>`: Average predictions across the 3 LOGO fold checkpoints (`models/{arch}_fold_{1,2,3}.pt`).
- `--hybrid-dead-ball`: Fuses physical stoppage and kinematic detection with neural emissions (+1.5 log-odds boost by default).
- `--dead-ball-boost <float>`: Additive log-score boost for frames flagged by the physical dead-ball detector (default: `1.5`).
- `--trans-penalty-ha <float>`: Viterbi transition log-penalty between Home and Away states (default: `3.0`).
- `--trans-penalty-dead <float>`: Viterbi transition log-penalty between Live Play and Dead Ball (default: `2.0`).
- `--abstain-threshold <float>`: Maximum probability threshold below which occluded ball frames emit `X` (default: `0.45`).
- `--no-abstain`: Disables selective abstention, emitting strictly `{H, A, D}`.

**Argument Rules (checked before any data is loaded):**
- `--checkpoints` and `--ensemble` are mutually exclusive.
- `--weights` requires `--checkpoints`, must have one finite non-negative value per checkpoint, and must sum to a positive value. Weights are normalized to sum to 1.
- `--stride` must satisfy `1 <= stride <= seq-len`, so every frame is covered by at least one window.
- Transition penalties and `--dead-ball-boost` must be finite and non-negative; `--abstain-threshold` must lie in `[0, 1]`.
- Missing checkpoint files are reported up front, all at once.

---

### 4.4 Validating and Scoring Output Submissions
Before submitting or evaluating any generated CSV, verify that it complies strictly with the competition submission contract:

```bash
python evaluation/validate_submission.py results/lamasia_helvetia.csv
```

The script verifies:
1. Exact row count matching the video horizon (199,319 rows).
2. Required CSV header: `frame,label`.
3. Contiguous integer sequence from `0` to `199318` in the `frame` column.
4. Absence of empty cells or `NaN` values.
5. Strict membership of labels in the valid alphabet: `{'H', 'A', 'D', 'X'}`.
6. Analyzes possession statistics (total spells, mean spell duration, and class balance).

To score the generated submission against the public 5-minute ground truth alongside the four official baselines (`baseline_*.csv`), copy the CSV into `evaluation/results/` and run the official scorer:

```bash
cp results/lamasia_helvetia.csv evaluation/results/
python evaluation/run_evaluation.py
```

`--results results` is required because the scorer looks in `evaluation/results/` by default, while the pipelines in this repository write to `results/` at the repository root.

> [!NOTE]
> **Operational & Reproducibility Notes:**
> - First Half (`11185..90651`) and Second Half (`110808..190197`) are processed independently, while pre-match, halftime (`90652..110807`), and post-match frames are deterministically assigned `D`.
> - Deterministic seeding and IEEE `float64` Viterbi path accumulation guarantee identical output across runs.
