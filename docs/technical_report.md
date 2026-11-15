# Technical Report: Ball Possession Identification
**More Than A Hack 2027 - Phase 1**  
**Team:** La MasIA Helvetia  
**Match Authority:** FC Barcelona vs Athletic Club (`BAR-ATH`) - 199,319 frames @ 25 fps (~132.9 min)  
**Alphabet:** Home (`H`), Away (`A`), Dead Ball (`D`), Selective Abstention (`X`)

---

## Executive Overview

This report details the ball possession identification methodology developed by team **La MasIA Helvetia** for Phase 1 of *More Than A Hack 2027*. The task requires continuous frame-by-frame possession segmentation across 199,319 frames at 25 fps, where the primary ranking metric is **$\text{spellF1}$** (temporal IoU matching of continuous possession intervals across thresholds 0.3–0.7) and the secondary tie-breaker is **$\text{macroF1}$** across $\{H, A, D\}$. 

To achieve high spell continuity under real-world tracking constraints (notably a ~41% ball occlusion rate and sensor teleportation spikes), we designed a **hybrid physics-informed sequence learning pipeline**. The system reconstructs ball trajectories using monotonic cubic interpolation, extracts 14 domain-invariant kinematic features, classifies states using deep sequence models (`PossessionBiGRU` and symmetric non-causal `PossessionTCN1D`) trained exclusively on external open tracking data, fuses physical dead-ball detections, and decodes the timeline via a Numba-accelerated 3-state Viterbi dynamic program with selective abstention (`X`).

```
  +-------------------------------------------------------------------------------------------------+
  |                                     INPUT ASSETS (BAR-ATH)                                      |
  |      Automatic Tracking Data (ATD)               Smart Tagging XML         EPTS Metadata XML    |
  +-------------------------------------------------------------------------------------------------+
                         |                                   |                         |
                         v                                   v                         v
  +--------------------------------------------+   +------------------------------------------------+
  |        BALL RECONSTRUCTION (PCHIP)         |   |          PHYSICAL DEAD-BALL DETECTOR           |
  |  - Kinematic spike filter (> 40 m/s)       |   |  - Official half boundaries (halftime masked)  |
  |  - Savitzky-Golay smoothing (win=7)        |   |  - Smart Tagging dilation (5.0 s window)       |
  |  - PCHIP interpolation (< 1.4 s occlusions)|   |  - Pitch boundary latching (margin = 0.6 m)    |
  |  - Blackouts (>= 1.4 s) flagged uncertain  |   |  - Outfield speed drop (p80 < 1.35-1.40 m/s)   |
  +--------------------------------------------+   +------------------------------------------------+
                         |                                                 |
                         +------------------------+------------------------+
                                                  v
  +--------------------------------------------------------------------------------------------------+
  |                       14 CANONICAL DOMAIN-INVARIANT KINEMATIC FEATURES                           |
  | Pitch-normalized ball (x, y), team distances & speeds, outfield tempo (mean/p80), team centroids |
  +--------------------------------------------------------------------------------------------------+
                                                  |
                         +------------------------+------------------------+
                         v                                                 v
  +--------------------------------------------+   +------------------------------------------------+
  |         SEQUENCE LEARNING MODELS           |   |          DETERMINISTIC HEURISTIC ENGINE        |
  |  - PossessionBiGRU (2-layer, residual MLP) |   |  - Pass-flight freezing (> 2.2 m free flight)  |
  |  - PossessionTCN1D (dilations 1..16, 10s RF|   |  - Scramble duel abstention (radius = 1.8 m)   |
  |  - Trained on external Metrica Games 1-3   |   |  - Micro-flip absorption (< 1.4 s live spells) |
  +--------------------------------------------+   +------------------------------------------------+
                         |                                                 |
                         v                                                 |
  +------------------------------------------------------------------+     |
  |               HYBRID KINEMATIC EMISSION FUSION                   |     |
  |  - Multi-model probability blending across architectures         |     |
  |  - Physical Dead-Ball log-odds boost (+1.5 on kinematic D)       |     |
  +------------------------------------------------------------------+     |
                         |                                                 |
                         v                                                 |
  +------------------------------------------------------------------+     |
  |                  NUMBA VITERBI SEQUENCE DECODER                  |     |
  |  - Transition penalties: H <-> A (3.0), Live <-> D (2.0)         |     |
  |  - Selective abstention ('X') on unobserved ball & low conf      |     |
  +------------------------------------------------------------------+     |
                         |                                                 |
                         +------------------------+------------------------+
                                                  v
  +-------------------------------------------------------------------------------------------------+
  |                                  SUBMISSION FILE & VALIDATION                                   |
  |                  199,319 rows (frame, label) verified via validate_submission.py                 |
  +-------------------------------------------------------------------------------------------------+
```

---

## 1. Assets Used & Assets Dropped

### Assets Used
- **Automatic Tracking Data (ATD) Tactical (`BAR-ATH_tactical_atd.txt`):** The primary quantitative signal, providing 2D metric coordinates on a $105.3\text{ m} \times 68.0\text{ m}$ pitch for all 22 player slots and the ball across all 199,319 video frames at 25 fps.
- **Smart Tagging XML (`BAR-ATH_tactical_smart_tagging.xml`):** Provides coarse tactical phase intervals, set-piece types (kickoff, throw-in, corner kick, free kick, goal kick, penalty), stoppage timestamps, and associated team attribution strings.
- **EPTS Metadata XML (`BAR-ATH_tactical_atd.xml`):** Used to extract official club identities (`idLocalTeam` = FC Barcelona, `idVisitingTeam` = Athletic Club) and isolate goalkeeper tracking slots (`position_index == 0`) from outfield tactical kinematics.

### Assets Dropped & Why
- **Broadcast & Tactical Video (`.mp4`):** *Dropped.* Computer vision models (e.g., YOLO or video action transformers) require significant GPU resources and introduce spatial calibration errors under dynamic broadcast zooming. Relying strictly on metric pitch coordinates guarantees zero video decoding overhead and pure domain invariance.
- **Homography Matrices (`_homography.csv`):** *Dropped.* The competition EPTS tracking data is already pre-calibrated and projected onto the 2D pitch plane in metric units. Image-to-pitch homography projection was unnecessary.
- **Tabular GBDT (`PossessionLGBM`):** *Dropped from the final operational blend.* While a multi-scale LightGBM model was implemented (`src/ml_pipeline/models/lgbm_model.py`) using 218 rolling temporal features, tree-based frame classifiers lack the continuous internal recurrent memory of neural sequence architectures (`BiGRU` / `TCN`). Under extended ball blackouts, GBDT predictions exhibited higher boundary chattering, reducing interval-level $\text{spellF1}$.

---

## 2. External Training Data & Zero-Leakage Compliance

### External Labelled Dataset
All supervised sequence models were trained exclusively on the **Metrica Sports Open Sample Data** (Sample Games 1, 2, and 3), accessed under an open research license:
- **Volume:** ~420,000 total frames at 25 fps across three complete professional matches.
- **Label Annotation:** Derived from Metrica Sports' synchronized event logs. Active event intervals (passes, carries, challenges, recoveries, shots) were mapped to team possession codes ($H = 0, A = 1$), while set pieces, out-of-bounds intervals, goals, and inter-half gaps were mapped to Dead Ball ($D = 2$).
- **Caching:** Ingested and precomputed into compressed archives (`external_open/.cache/*.npz`), enabling instant (< 0.2 s) data loading during experiments.

### Zero-Leakage Governance
- **Strict Match Data Isolation:** Absolutely **zero frames, labels, or video** from the competition match (`BAR-ATH`) were used to train network weights or adjust heuristic thresholds.
- **No Ground-Truth Snooping:** The public 5-minute ground-truth slice was used solely as an offline format verification check, never as an optimization objective.
- **Domain Invariance:** Distances, speeds, and formation centroids are projected into pitch-normalized metric coordinates, ensuring models generalize seamlessly across stadiums.

### Domain-Aligned Training Augmentations (`src/ml_pipeline/dataset_open.py`)
To bridge the gap between clean open-data tracking and the noisy competition environment, three on-the-fly augmentations were applied during training:
1. **Synthetic 41% Ball Occlusion Dropout & Jitter:** Randomly masks contiguous ball observation spans (5–75 frames, 0.2–3.0 s) to match the competition missingness rate, while adding Gaussian coordinate jitter ($\sigma = 0.25\text{ m}$) on visible frames.
2. **Symmetrical Channel & Label Swapping ($p = 0.5$):** Symmetrically transposes Home and Away distances, closest player speeds, and team centroids while swapping ground-truth labels ($0 \leftrightarrow 1$), eliminating home-team bias.
3. **Pitch Coordinate Reflection ($p = 0.5$):** Randomly mirrors $X$ (longitudinal) and $Y$ (lateral) coordinates to remove tactical directionality bias.

---

## 3. Methodology & Possession Rules

### 3.1 Ball Reconstruction & Dead-Ball Detection

#### Ball Trajectory Reconstruction (`src/ball_tracking.py`)
1. **Kinematic Spike Filtering:** Optical tracking occasionally introduces 1–3 frame teleportation spikes where the ball jumps to a stray artifact. A bidirectional reachability check removes intermediate detections requiring speeds $> 40.0\text{ m/s}$ (~144 km/h) if the bounding anchors are mutually reachable.
2. **Savitzky-Golay Smoothing:** Contiguous valid observation segments are smoothed (`window=7`, `polyorder=2`) to eliminate sensor jitter prior to velocity estimation.
3. **Monotonic PCHIP Interpolation:** Gaps $< 35$ frames ($1.4\text{ s}$) with average speed $\le 38.0\text{ m/s}$ are reconstructed using monotonic cubic Hermite polynomials (`PchipInterpolator`), with instantaneous velocities computed via exact analytic derivatives. Gaps $\ge 35$ frames are flagged as unobserved blackouts (`is_ball_uncertain = True`).

#### Physical Dead-Ball Detection (`src/dead_ball.py`)
1. **Playing Period Masking:** Halftime (20,156 frames), pre-match, and post-match spans are locked to Dead Ball ($D$) based on official half boundaries.
2. **Smart Tagging Dilation:** Stoppage and set-piece intervals are expanded with a 125-frame ($5.0\text{ s}$) binary dilation to capture the setup phase preceding the tagged whistle.
3. **Out-of-Bounds Latching:** When the ball exceeds pitch boundaries ($\pm 0.6\text{ m}$ margin), Dead Ball is latched forward across subsequent frames until active controlled play re-enters the pitch (ball in-bounds with a player within $2.2\text{ m}$) or the stoppage event concludes.
4. **Collective Outfield Kinematics ($p_{80}$):** When the 80th-percentile outfield player speed drops below $1.40\text{ m/s}$ for $\ge 2.0\text{ s}$ during a dilated stoppage or abandoned ball ($> 2.5\text{ m}$ from all players), Dead Ball is triggered. Standalone stoppages trigger when $p_{80} < 1.35\text{ m/s}$ for $\ge 2.4\text{ s}$, catching untagged referee pauses.

---

### 3.2 Possession Boundaries: Loose Balls, Challenges, Restarts & 'X'

| Tactical Scenario | Physical Condition | Model Operational Rule | Output Label |
|:---|:---|:---|:---:|
| **Pass Flight / Loose Ball** | Ball in free flight ($> 2.2\text{ m}$ from all players) | Possession turnover evidence (`opp_evidence`) is **frozen**. Possession does not flip while the ball is traveling to a teammate. | Retains $H$ or $A$ |
| **Occluded Dribble** | Ball occluded, possession established, model confidence $\ge 0.45$ | Possession continuity is maintained through body-shielding occlusions based on player movement. | Retains $H$ or $A$ |
| **Contested Duel / Scramble** | Both teams $\le 1.8\text{ m}$ from ball, distance gap $< 0.4\text{ m}$ for $\ge 15$ frames | Possession becomes unestablished; engine refrains from guessing during actively contested scrambles. | Emits $X$ |
| **Low-Confidence Blackout** | Ball occluded ($> 1.4\text{ s}$) and model max probability $< 0.45$ | Selective abstention is triggered to prevent inserting spurious transitions during unresolvable tracking loss. | Emits $X$ |
| **Set-Piece Restart** | Play transitions from $D \to \text{Live}$ | Restarting team is anchored to Smart Tagging metadata or the first player reaching within $2.5\text{ m}$ within 50 frames. | Begins $H$ or $A$ spell |
| **Out-of-Play / Stoppage** | Whistle blown, ball out-of-bounds, or collective $p_{80} < 1.35\text{ m/s}$ | Kinematic stoppage detection overrides live-play classifications. | Emits $D$ |

---

### 3.3 Sequence Modeling & Hybrid Viterbi Decoding

#### Canonical 14-Dimensional Feature Vector (`src/ml_pipeline/features.py`)
Every frame is projected into a standardized feature vector:

| Feature | Name | Semantic Definition |
|:---:|:---|:---|
| $0, 1$ | `min_dist_home`, `min_dist_away` | Distance (m) from ball to nearest Home / Away outfield player (sentinel $50.0\text{ m}$ if occluded) |
| $2$ | `dist_diff` | Proximity advantage: $\text{min\_dist\_away} - \text{min\_dist\_home}$ ($0.0$ if occluded) |
| $3, 4$ | `is_ball_visible`, `ball_speed` | Ball visibility indicator ($0/1$) and estimated ball speed ($\text{m/s}$) |
| $5, 6$ | `ball_x_norm`, `ball_y_norm` | Pitch-normalized ball coordinates relative to center in $[-1.0, 1.0]$ |
| $7, 8$ | `closest_home_spd`, `closest_away_spd` | Speeds ($\text{m/s}$) of the nearest Home and Away outfield players to the ball |
| $9, 10$| `outfield_mean_spd`, `outfield_p80_spd` | Collective outfield player speed statistics ($\text{m/s}$) |
| $11, 12$| `home_cx_norm`, `away_cx_norm` | Normalized longitudinal formation centroids in $[-1.0, 1.0]$ |
| $13$ | `is_stoppage_tag` | Binary indicator for Smart Tagging stoppage and set-piece intervals |

#### Neural Sequence Architectures (`src/ml_pipeline/models/`)
1. **Bidirectional GRU (`PossessionBiGRU`):** Two stacked bidirectional GRU layers ($d = 96$, combined 192) with an input LayerNorm projection and a residual MLP classification head. Regularized during training with an explicit temporal smoothness penalty ($\lambda = 0.1$) on consecutive frame logits to discourage high-frequency output oscillation.
2. **Symmetric Non-Causal 1D-TCN (`PossessionTCN1D`):** Multi-scale 1D dilated temporal convolutional network using dilation cycles $(1, 2, 4, 8, 16)$ with GroupNorm and GELU activations. Its symmetric bidirectional receptive field (~250 frames / 10.0 s) localizes possession transitions with zero temporal phase lag.

#### Hybrid Kinematic Fusion & Viterbi Sequence Decoding
- **Kinematic Dead-Ball Boosting (`--hybrid-dead-ball`):** Multi-model ensemble probabilities are converted to log-odds. For frames confirmed as Dead Ball by physical and speed heuristics, Dead Ball emission log-probabilities receive an additive boost of $+1.5$, locking confirmed stoppages.
- **Numba-Accelerated Viterbi Decoder (`src/ml_pipeline/viterbi.py`):** Solves the global optimal state path across each half independently in $< 5\text{ ms}$ using JIT-compiled dynamic programming. Transition penalties suppress spurious micro-flips ($H \leftrightarrow A$ penalty $= 3.0$, Live $\leftrightarrow D$ penalty $= 2.0$).
- **Selective Abstention (`X`):** Applied conditionally post-Viterbi when the ball is unobserved and model confidence is below $0.45$.

---

## 4. Limitations & Future Improvements

1. **2D Pitch Projection Limits (Lack of Z-Axis Elevation):**
   - *Limitation:* Optical ATD provides only 2D ground coordinates $(x, y)$. High aerial clearances, crosses, and contested aerial duels can appear proximate to ground defenders who have no physical play on the ball.
   - *Improvement:* Estimate 3D ball parabolas from video camera homographies or player bounding-box heights to explicitly detect aerial flight phases.
2. **Prolonged Ball Blackouts ($\ge 1.4\text{ s}$):**
   - *Limitation:* During multi-second tracking losses where the ball leaves the camera view, the system must rely entirely on outfield formation drift and collective player speeds.
   - *Improvement:* Implement a secondary optical ball detector trained on broadcast video cuts to re-anchor ball position during tactical camera occlusions.
3. **Slow Set-Piece Restart Latency:**
   - *Limitation:* In throw-ins or free kicks where players walk slowly toward the ball prior to restarting play, identifying the exact transition frame from $D \to \text{Live}$ can exhibit a 5–10 frame latency if Smart Tagging timestamps are imprecise.
   - *Improvement:* Incorporate player head-pose and kicking-action detection from tactical video to isolate the exact striking instant.
4. **External Training Corpus Size:**
   - *Limitation:* Training was restricted to 3 open Metrica matches (~420k frames), limiting tactical diversity to specific team styles.
   - *Improvement:* Pre-train sequence backbones on large-scale synthetic match simulations or additional multi-league open datasets.

---

## 5. Reproduction Instructions

### 5.1 Environment Setup
```bash
# Clone repository and enter environment
cd /workspaces/barca-hackathon

# Install pinned dependencies (PyTorch CPU, Numba, LightGBM, Kloppy)
pip install -r requirements.txt
```

### 5.2 Match Inference (`BAR-ATH`)
Ensure match assets are located in `./data/tactical/` or set `export MTAH_DATA_DIR=/path/to/data`.

```bash
# Workflow A: Single Model Inference (BiGRU Full Checkpoint)
python src/ml_pipeline/infer_match.py \
  --checkpoint models/bi_gru_full.pt \
  --hybrid-dead-ball \
  --output results/lamasia_helvetia_bigru.csv

# Workflow B: Heterogeneous Neural Ensemble (BiGRU + TCN)
python src/ml_pipeline/infer_match.py \
  --checkpoints models/bi_gru_full.pt models/tcn_1d_full.pt \
  --weights 0.60 0.40 \
  --hybrid-dead-ball \
  --output results/lamasia_helvetia_ensemble.csv

# Workflow C: Deterministic Physics/Heuristic Pipeline (No Model Weights Needed)
python src/run_pipeline.py \
  --data-dir ./data \
  --output results/lamasia_helvetia_physics.csv
```

### 5.3 Submission Validation
Verify compliance against the official competition submission contract:
```bash
python evaluation/validate_submission.py results/lamasia_helvetia_bigru.csv
```

---

## 6. Experimental Validation & Final Model Selection

_To be completed with external Leave-One-Game-Out validation summary and final model selection._
