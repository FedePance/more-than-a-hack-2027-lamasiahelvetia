"""Viterbi sequence decoding and selective abstention for possession states.

Decodes frame-level class logits/probabilities into a globally consistent state
sequence (0=H, 1=A, 2=D, 3=X) by:
1. Penalizing rapid spurious transitions between opposing teams (H <-> A) and
   between active play and stoppage (live <-> D).
2. Numba JIT acceleration: forward DP and backtracking executed in C-speed (< 5 ms for 150k frames).
3. Selective abstention ('X'): Emits 'X' only when the ball is unobserved / occluded
   AND the model confidence (max state probability) is below a confidence threshold.
"""

import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

CODE_TO_LABEL = {0: "H", 1: "A", 2: "D", 3: "X"}
LABEL_TO_CODE = {"H": 0, "A": 1, "D": 2, "X": 3}


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along last axis."""
    shift = logits - np.max(logits, axis=-1, keepdims=True)
    exp_vals = np.exp(shift)
    return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax along last axis."""
    shift = logits - np.max(logits, axis=-1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(shift), axis=-1, keepdims=True))
    return shift - log_sum_exp


if HAS_NUMBA:
    @njit(fastmath=True)
    def _viterbi_dp_numba(
        log_probs: np.ndarray,
        log_trans: np.ndarray,
        prior: np.ndarray,
    ) -> np.ndarray:
        """Numba-accelerated forward DP and backtracking."""
        n_frames, n_states = log_probs.shape
        viterbi = np.zeros((n_frames, n_states), dtype=np.float32)
        backpointers = np.zeros((n_frames, n_states), dtype=np.int32)
        viterbi[0] = prior + log_probs[0]

        for t in range(1, n_frames):
            for s in range(n_states):
                max_val = -1e30
                best_prev = 0
                for prev_s in range(n_states):
                    score = viterbi[t - 1, prev_s] + log_trans[prev_s, s]
                    if score > max_val:
                        max_val = score
                        best_prev = prev_s
                backpointers[t, s] = best_prev
                viterbi[t, s] = max_val + log_probs[t, s]

        decoded = np.zeros(n_frames, dtype=np.int8)
        best_last = 0
        max_last = viterbi[n_frames - 1, 0]
        for s in range(1, n_states):
            if viterbi[n_frames - 1, s] > max_last:
                max_last = viterbi[n_frames - 1, s]
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
    """Pure NumPy forward DP and backtracking fallback."""
    n_frames, n_states = log_probs.shape
    viterbi = np.zeros((n_frames, n_states), dtype=np.float32)
    backpointers = np.zeros((n_frames, n_states), dtype=np.int32)
    viterbi[0] = prior + log_probs[0]

    for t in range(1, n_frames):
        candidate_scores = viterbi[t - 1, :, None] + log_trans
        best_prev = np.argmax(candidate_scores, axis=0)
        backpointers[t] = best_prev
        viterbi[t] = candidate_scores[best_prev, np.arange(n_states)] + log_probs[t]

    decoded = np.zeros(n_frames, dtype=np.int8)
    decoded[-1] = np.argmax(viterbi[-1])

    for t in range(n_frames - 2, -1, -1):
        decoded[t] = backpointers[t + 1, decoded[t + 1]]

    return decoded


def viterbi_decode(
    emissions: np.ndarray,
    is_ball_visible: np.ndarray | None = None,
    trans_penalty_ha: float = 3.0,
    trans_penalty_dead: float = 2.0,
    abstain_threshold: float = 0.45,
    enable_abstention: bool = True,
    prior: np.ndarray | None = None,
) -> np.ndarray:
    """Decode per-frame logits/probabilities into globally optimal possession codes.

    States:
        0 = Home ('H')
        1 = Away ('A')
        2 = Dead Ball ('D')
        3 = Abstain ('X')

    Args:
        emissions: Array of shape (n_frames, 3) representing logits, log-probabilities,
            or raw probabilities for states (H, A, D).
        is_ball_visible: Boolean or float array of shape (n_frames,), where 0/False
            indicates missing/occluded ball. Required if enable_abstention is True.
        trans_penalty_ha: Transition log-penalty for changing between H and A (default: 3.0).
        trans_penalty_dead: Transition log-penalty for entering/exiting Dead Ball D (default: 2.0).
        abstain_threshold: Confidence threshold below which unobserved ball frames abstain (default: 0.45).
        enable_abstention: Whether to emit 'X' during low-confidence blackouts (default: True).
        prior: Optional initial state log-probabilities, shape (3,).

    Returns:
        np.ndarray of shape (n_frames,), dtype int8 with decoded codes in {0, 1, 2, 3}.
    """
    n_frames, n_states = emissions.shape
    if n_states != 3:
        raise ValueError(f"Expected emissions with 3 states (H, A, D), got {n_states}")

    if n_frames == 0:
        return np.zeros(0, dtype=np.int8)

    # Convert to log-probabilities and raw probabilities
    if np.any(emissions < 0.0) or np.any(emissions > 1.0) or not np.allclose(emissions.sum(axis=-1), 1.0, atol=1e-2):
        log_probs = _log_softmax(emissions).astype(np.float32)
        probs = _softmax(emissions).astype(np.float32)
    else:
        probs = emissions.astype(np.float32)
        log_probs = np.log(np.clip(emissions, 1e-12, 1.0)).astype(np.float32)

    # Transition log-cost matrix
    # State order: 0=H, 1=A, 2=D
    log_trans = np.array(
        [
            [0.0, -trans_penalty_ha, -trans_penalty_dead],
            [-trans_penalty_ha, 0.0, -trans_penalty_dead],
            [-trans_penalty_dead, -trans_penalty_dead, 0.0],
        ],
        dtype=np.float32,
    )

    prior_vec = np.zeros(n_states, dtype=np.float32) if prior is None else prior.astype(np.float32)

    # Forward DP and backtracking
    if HAS_NUMBA:
        decoded = _viterbi_dp_numba(log_probs, log_trans, prior_vec)
    else:
        decoded = _viterbi_dp_numpy(log_probs, log_trans, prior_vec)

    # Selective abstention ('X' = 3)
    if enable_abstention and is_ball_visible is not None:
        vis_mask = np.asarray(is_ball_visible, dtype=bool)
        max_prob = np.max(probs, axis=-1)
        abstain_mask = (~vis_mask) & (max_prob < abstain_threshold)
        decoded[abstain_mask] = 3

    return decoded


def codes_to_labels(codes: np.ndarray) -> np.ndarray:
    """Convert integer state codes {0, 1, 2, 3} to character labels {'H', 'A', 'D', 'X'}."""
    label_map = np.array(["H", "A", "D", "X"], dtype=object)
    return label_map[codes]
