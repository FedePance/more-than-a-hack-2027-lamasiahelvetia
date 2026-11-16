"""Viterbi sequence decoding and selective abstention for possession states.

Decodes frame-level class logits/probabilities into a globally consistent state
sequence (0=H, 1=A, 2=D, 3=X) by:
1. Penalizing rapid spurious transitions between opposing teams (H <-> A) and
   between active play and stoppage (live <-> D).
2. Numba JIT acceleration: forward DP and backtracking executed in C-speed (< 5 ms for 150k frames).
3. Selective abstention ('X'): Emits 'X' only when the ball is unobserved / occluded
   AND the model confidence (max state probability) is below a confidence threshold.

Determinism: path scores are accumulated in float64 with strict IEEE semantics (no fastmath),
and ties are always resolved toward the lowest state index, so the Numba kernel and the NumPy
fallback return identical paths on every platform.
"""

import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

CODE_TO_LABEL = {0: "H", 1: "A", 2: "D", 3: "X"}
LABEL_TO_CODE = {"H": 0, "A": 1, "D": 2, "X": 3}

N_STATES = 3
ABSTAIN_CODE = LABEL_TO_CODE["X"]
EMISSION_TYPES = ("auto", "logits", "probs")

# Probability floor applied before taking logs, and bound on raw logit magnitude
_MIN_PROB = 1e-12
_MAX_ABS_LOGIT = 1e4

_LABEL_LOOKUP = np.array(["H", "A", "D", "X"], dtype=object)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along last axis."""
    shift = logits - np.max(logits, axis=-1, keepdims=True)
    exp_vals = np.exp(shift)
    return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)


def log_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax along last axis."""
    shift = logits - np.max(logits, axis=-1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(shift), axis=-1, keepdims=True))
    return shift - log_sum_exp


if HAS_NUMBA:
    @njit
    def _viterbi_dp_numba(
        log_probs: np.ndarray,
        log_trans: np.ndarray,
        prior: np.ndarray,
    ) -> np.ndarray:
        """Numba-accelerated forward DP and backtracking (float64 C-contiguous inputs)."""
        n_frames, n_states = log_probs.shape
        scores = np.zeros((n_frames, n_states), dtype=np.float64)
        backpointers = np.zeros((n_frames, n_states), dtype=np.int8)
        for s in range(n_states):
            scores[0, s] = prior[s] + log_probs[0, s]

        for t in range(1, n_frames):
            for s in range(n_states):
                best_score = scores[t - 1, 0] + log_trans[0, s]
                best_prev = 0
                for prev_s in range(1, n_states):
                    score = scores[t - 1, prev_s] + log_trans[prev_s, s]
                    if score > best_score:
                        best_score = score
                        best_prev = prev_s
                backpointers[t, s] = best_prev
                scores[t, s] = best_score + log_probs[t, s]

        decoded = np.zeros(n_frames, dtype=np.int8)
        best_last = 0
        for s in range(1, n_states):
            if scores[n_frames - 1, s] > scores[n_frames - 1, best_last]:
                best_last = s
        decoded[n_frames - 1] = best_last

        for t in range(n_frames - 2, -1, -1):
            decoded[t] = backpointers[t + 1, decoded[t + 1]]

        return decoded


def _viterbi_dp_numpy(
    log_probs: np.ndarray,
    log_trans: np.ndarray,
    prior: np.ndarray,
) -> np.ndarray:
    """Pure NumPy forward DP and backtracking fallback (same arithmetic as the Numba kernel)."""
    n_frames, n_states = log_probs.shape
    scores = np.zeros((n_frames, n_states), dtype=np.float64)
    backpointers = np.zeros((n_frames, n_states), dtype=np.int8)
    scores[0] = prior + log_probs[0]
    state_range = np.arange(n_states)

    for t in range(1, n_frames):
        candidate_scores = scores[t - 1, :, None] + log_trans
        best_prev = np.argmax(candidate_scores, axis=0)
        backpointers[t] = best_prev
        scores[t] = candidate_scores[best_prev, state_range] + log_probs[t]

    decoded = np.zeros(n_frames, dtype=np.int8)
    decoded[-1] = np.argmax(scores[-1])

    for t in range(n_frames - 2, -1, -1):
        decoded[t] = backpointers[t + 1, decoded[t + 1]]

    return decoded


def _looks_like_probabilities(emissions: np.ndarray) -> bool:
    """Heuristic used by emission_type='auto': every row is a distribution over the states."""
    return bool(
        not np.any(emissions < 0.0)
        and not np.any(emissions > 1.0)
        and np.allclose(emissions.sum(axis=-1), 1.0, atol=1e-2)
    )


def _emission_log_probs(emissions: np.ndarray, emission_type: str) -> tuple[np.ndarray, np.ndarray]:
    """Convert raw emissions to (log_probs, probs), both float32 and finite.

    Non-finite inputs never reach the DP: NaN rows become uniform, infinite logits are clamped
    to +/- 1e4, and probabilities are floored at 1e-12 before the log.
    """
    emissions = np.asarray(emissions)
    if not np.issubdtype(emissions.dtype, np.floating):
        emissions = emissions.astype(np.float32)

    is_finite = bool(np.isfinite(emissions).all())
    if emission_type == "auto":
        as_probs = is_finite and _looks_like_probabilities(emissions)
    else:
        as_probs = emission_type == "probs"

    if as_probs:
        if not is_finite:
            emissions = np.clip(np.nan_to_num(emissions, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
            emissions[emissions.sum(axis=-1) <= 0.0] = 1.0 / emissions.shape[-1]
        probs = emissions.astype(np.float32)
        log_probs = np.log(np.clip(emissions, _MIN_PROB, 1.0)).astype(np.float32)
        return log_probs, probs

    if not is_finite:
        nan_rows = np.isnan(emissions).any(axis=-1)
        emissions = np.clip(emissions, -_MAX_ABS_LOGIT, _MAX_ABS_LOGIT)
        emissions[nan_rows] = 0.0
    log_probs = log_softmax(emissions).astype(np.float32)
    probs = softmax(emissions).astype(np.float32)
    return log_probs, probs


def viterbi_decode(
    emissions: np.ndarray,
    is_ball_visible: np.ndarray | None = None,
    trans_penalty_ha: float = 3.0,
    trans_penalty_dead: float = 2.0,
    abstain_threshold: float = 0.45,
    enable_abstention: bool = True,
    prior: np.ndarray | None = None,
    emission_type: str = "auto",
) -> np.ndarray:
    """Decode per-frame logits/probabilities into globally optimal possession codes.

    States:
        0 = Home ('H')
        1 = Away ('A')
        2 = Dead Ball ('D')
        3 = Abstain ('X')

    Args:
        emissions: Array of shape (n_frames, 3) with per-frame scores for states (H, A, D).
        is_ball_visible: Boolean or float array of shape (n_frames,), where 0/False
            indicates missing/occluded ball. Abstention is skipped when it is None.
        trans_penalty_ha: Transition log-penalty for changing between H and A (default: 3.0).
        trans_penalty_dead: Transition log-penalty for entering/exiting Dead Ball D (default: 2.0).
        abstain_threshold: Confidence threshold below which unobserved ball frames abstain (default: 0.45).
        enable_abstention: Whether to emit 'X' during low-confidence blackouts (default: True).
        prior: Optional initial state log-probabilities, shape (3,).
        emission_type: How to interpret emissions:
            'logits' - unnormalized log-scores (logits, log-probabilities, or log-probabilities
                       with additive evidence); normalized with a log-softmax.
            'probs'  - probabilities in [0, 1].
            'auto'   - 'probs' if every row is a valid distribution, otherwise 'logits'.

    Returns:
        np.ndarray of shape (n_frames,), dtype int8 with decoded codes in {0, 1, 2, 3}.

    Raises:
        ValueError: On malformed shapes, an unknown emission_type, or negative / non-finite penalties.
    """
    emissions = np.asarray(emissions)
    if emissions.ndim != 2 or emissions.shape[1] != N_STATES:
        raise ValueError(f"Expected emissions of shape (n_frames, {N_STATES}) for states (H, A, D), got {emissions.shape}")
    if emission_type not in EMISSION_TYPES:
        raise ValueError(f"Unknown emission_type '{emission_type}'. Available options: {EMISSION_TYPES}")
    for name, penalty in (("trans_penalty_ha", trans_penalty_ha), ("trans_penalty_dead", trans_penalty_dead)):
        if not np.isfinite(penalty) or penalty < 0.0:
            raise ValueError(f"{name} must be a finite non-negative number, got {penalty}")

    n_frames = emissions.shape[0]
    if n_frames == 0:
        return np.zeros(0, dtype=np.int8)

    vis_mask = None
    if enable_abstention and is_ball_visible is not None:
        vis_mask = np.asarray(is_ball_visible)
        if vis_mask.shape != (n_frames,):
            raise ValueError(f"is_ball_visible must have shape ({n_frames},), got {vis_mask.shape}")
        vis_mask = vis_mask.astype(bool)

    log_probs, probs = _emission_log_probs(emissions, emission_type)

    # Transition log-cost matrix
    # State order: 0=H, 1=A, 2=D
    log_trans = np.array(
        [
            [0.0, -trans_penalty_ha, -trans_penalty_dead],
            [-trans_penalty_ha, 0.0, -trans_penalty_dead],
            [-trans_penalty_dead, -trans_penalty_dead, 0.0],
        ],
        dtype=np.float64,
    )

    if prior is None:
        prior_vec = np.zeros(N_STATES, dtype=np.float64)
    else:
        prior_vec = np.ascontiguousarray(prior, dtype=np.float64)
        if prior_vec.shape != (N_STATES,) or np.isnan(prior_vec).any():
            raise ValueError(f"prior must hold {N_STATES} non-NaN log-probabilities, got {prior!r}")
        prior_vec = np.maximum(prior_vec, -_MAX_ABS_LOGIT)  # keep log(0) priors finite

    # Forward DP and backtracking in float64: with float32 the running path score loses
    # resolution over a ~80k-frame half and near-ties are resolved by rounding noise
    dp_log_probs = np.ascontiguousarray(log_probs, dtype=np.float64)
    if HAS_NUMBA:
        decoded = _viterbi_dp_numba(dp_log_probs, log_trans, prior_vec)
    else:
        decoded = _viterbi_dp_numpy(dp_log_probs, log_trans, prior_vec)

    # Selective abstention ('X' = 3)
    if vis_mask is not None:
        max_prob = np.max(probs, axis=-1)
        abstain_mask = (~vis_mask) & (max_prob < abstain_threshold)
        decoded[abstain_mask] = ABSTAIN_CODE

    return decoded


def codes_to_labels(codes: np.ndarray) -> np.ndarray:
    """Convert integer state codes {0, 1, 2, 3} to character labels {'H', 'A', 'D', 'X'}.

    Raises:
        ValueError: If any code falls outside {0, 1, 2, 3}, so an invalid label can never be written.
    """
    codes = np.asarray(codes)
    if not np.issubdtype(codes.dtype, np.integer):
        raise ValueError(f"State codes must be integers, got dtype {codes.dtype}")
    if codes.size and (codes.min() < 0 or codes.max() >= len(_LABEL_LOOKUP)):
        raise ValueError(
            f"State codes must lie in 0..{len(_LABEL_LOOKUP) - 1}, got range {codes.min()}..{codes.max()}"
        )
    return _LABEL_LOOKUP[codes]
