from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .tasks import Example, FormalTask, _final_after_marker, _prefix_before_final

_MATCH, _SKIP_ORACLE, _SKIP_GEN = 0, 1, 2


@dataclass
class StepScore:
    gen_index: Optional[int]
    oracle_index: Optional[int]
    components: Dict[str, float]
    total: float


@dataclass
class DenseRewardResult:
    step_scores: List[StepScore]
    token_rewards: List[float]
    terminal_reward: float
    sequence_process_reward: float
    dense_process_reward: float
    mean_step_reward: float
    num_correct_steps: int
    num_generated_steps: int
    num_oracle_steps: int
    step_alignment_accuracy: float
    total_reward: float


def _chunk(tokens: Sequence[str], size: int) -> List[Tuple[str, ...]]:
    n_full = len(tokens) // size
    return [tuple(tokens[i : i + size]) for i in range(0, n_full * size, size)]


def align_steps(
    oracle_steps: Sequence[Tuple[str, ...]],
    gen_steps: Sequence[Tuple[str, ...]],
    match_score_fn,
) -> List[Tuple[Optional[int], Optional[int]]]:
    """Monotonic global alignment maximizing summed step-similarity (Needleman-Wunsch style).

    Robust to missing oracle steps (dropped/skipped in generation), extra
    generated steps, and minor formatting issues -- a bad step only costs
    that one step's credit instead of desynchronizing every later comparison
    the way a naive positional zip does.
    """
    n, m = len(oracle_steps), len(gen_steps)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    back = [[_MATCH] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0]
        back[i][0] = _SKIP_ORACLE
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1]
        back[0][j] = _SKIP_GEN

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match_val = dp[i - 1][j - 1] + match_score_fn(oracle_steps[i - 1], gen_steps[j - 1])
            skip_o = dp[i - 1][j]
            skip_g = dp[i][j - 1]
            best, move = match_val, _MATCH
            if skip_o > best:
                best, move = skip_o, _SKIP_ORACLE
            if skip_g > best:
                best, move = skip_g, _SKIP_GEN
            dp[i][j] = best
            back[i][j] = move

    pairs: List[Tuple[Optional[int], Optional[int]]] = []
    i, j = n, m
    while i > 0 or j > 0:
        move = back[i][j]
        if move == _MATCH and i > 0 and j > 0:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif move == _SKIP_ORACLE and i > 0:
            pairs.append((i - 1, None))
            i -= 1
        else:
            pairs.append((None, j - 1))
            j -= 1
    pairs.reverse()
    return pairs


def _score_step(
    task: FormalTask,
    oracle_prev_state: Any,
    gen_prev_state: Optional[Any],
    symbol: str,
    oracle_step: Tuple[str, ...],
    gen_step: Tuple[str, ...],
    oracle_index: int,
    gen_index: int,
    partial_credit: bool,
) -> StepScore:
    if not partial_credit:
        total = task.step_match_score(oracle_step, gen_step)
        return StepScore(gen_index, oracle_index, {"match": total}, total)

    symbol_component = 1.0 if gen_index == oracle_index else 0.0

    if oracle_index == 0:
        prev_state_component = 1.0
    elif gen_prev_state is not None:
        prev_state_component = 1.0 if gen_prev_state == oracle_prev_state else 0.0
    else:
        prev_state_component = 0.0

    believed_prev = gen_prev_state if gen_prev_state is not None else task.initial_state()
    predicted_tokens = task.state_to_step_tokens(task.apply_transition(believed_prev, symbol))
    transition_component = 1.0 if predicted_tokens == tuple(gen_step) else 0.0

    next_state_component = task.step_match_score(oracle_step, gen_step)

    components = {
        "symbol": symbol_component,
        "prev_state": prev_state_component,
        "transition": transition_component,
        "next_state": next_state_component,
    }
    total = sum(components.values()) / len(components)
    return StepScore(gen_index, oracle_index, components, total)


def compute_dense_reward(
    task: FormalTask,
    example: Example,
    raw_generated_tokens: Sequence[str],
    *,
    terminal_weight: float = 1.0,
    process_weight: float = 1.0,
    dense_step_weight: float = 1.0,
    partial_credit: bool = True,
    normalize_dense_reward: bool = True,
) -> DenseRewardResult:
    """Compute per-step partial-credit rewards and align them back onto token positions.

    ``raw_generated_tokens`` must correspond 1:1 with the sampled token ids of a
    rollout (unfiltered apart from an optional trailing ``<eos>``), so the
    returned ``token_rewards`` can be used directly as a per-token reward
    vector for PPO/GRPO.
    """
    tokens = list(raw_generated_tokens)
    trailing_eos = bool(tokens) and tokens[-1] == "<eos>"
    core_tokens = tokens[:-1] if trailing_eos else tokens

    step_size = task.step_size()
    gen_prefix = _prefix_before_final(core_tokens)
    oracle_steps = _chunk(example.trace_tokens, step_size)
    gen_steps = _chunk(gen_prefix, step_size)

    alignment = align_steps(oracle_steps, gen_steps, task.step_match_score)

    oracle_state_history: List[Any] = [task.initial_state()]
    state = task.initial_state()
    for symbol in example.input_tokens:
        state = task.apply_transition(state, symbol)
        oracle_state_history.append(state)

    step_scores: List[StepScore] = []
    gen_reward_by_index: Dict[int, float] = {}
    last_gen_state: Optional[Any] = None
    correct_steps = 0
    matched_steps = 0

    for oracle_idx, gen_idx in alignment:
        if oracle_idx is None:
            continue  # extra generated step with no oracle counterpart

        if gen_idx is None:
            step_scores.append(StepScore(None, oracle_idx, {}, 0.0))
            continue

        symbol = example.input_tokens[oracle_idx] if oracle_idx < len(example.input_tokens) else ""
        score = _score_step(
            task,
            oracle_state_history[oracle_idx],
            last_gen_state,
            symbol,
            oracle_steps[oracle_idx],
            gen_steps[gen_idx],
            oracle_idx,
            gen_idx,
            partial_credit,
        )
        step_scores.append(score)
        gen_reward_by_index[gen_idx] = score.total
        matched_steps += 1
        if score.total >= 0.999:
            correct_steps += 1
        last_gen_state = task.parse_step_state(gen_steps[gen_idx])

    num_oracle_steps = len(oracle_steps)
    num_generated_steps = len(gen_steps)

    matched_total = sum(s.total for s in step_scores)
    dense_process_reward = matched_total / max(1, num_oracle_steps)
    mean_step_reward = matched_total / max(1, matched_steps) if matched_steps else 0.0
    step_alignment_accuracy = matched_steps / max(1, num_oracle_steps)

    legacy = task.reward(example, core_tokens)
    sequence_process_reward = legacy.process
    terminal_reward = legacy.terminal

    token_rewards = [0.0] * len(gen_prefix)
    for gen_idx, step_reward in gen_reward_by_index.items():
        start = gen_idx * step_size
        for offset in range(step_size):
            token_rewards[start + offset] = step_reward * dense_step_weight

    remainder = core_tokens[len(gen_prefix) :]
    remainder_rewards = [0.0] * len(remainder)
    if "FINAL" in remainder:
        final_pos = remainder.index("FINAL")
        for offset in (final_pos, final_pos + 1):
            if offset < len(remainder_rewards):
                remainder_rewards[offset] = terminal_reward * terminal_weight

    full_rewards = token_rewards + remainder_rewards
    if trailing_eos:
        full_rewards.append(0.0)

    process_component = dense_process_reward if normalize_dense_reward else matched_total
    total_reward = process_weight * process_component + terminal_weight * terminal_reward

    return DenseRewardResult(
        step_scores=step_scores,
        token_rewards=full_rewards,
        terminal_reward=terminal_reward,
        sequence_process_reward=sequence_process_reward,
        dense_process_reward=dense_process_reward,
        mean_step_reward=mean_step_reward,
        num_correct_steps=correct_steps,
        num_generated_steps=num_generated_steps,
        num_oracle_steps=num_oracle_steps,
        step_alignment_accuracy=step_alignment_accuracy,
        total_reward=total_reward,
    )


def prefix_accuracy(example: Example, generated: Sequence[str], step_size: int) -> float:
    """Fraction of steps, counted from the start, that are correct before the first mismatch."""
    oracle_steps = _chunk(example.trace_tokens, step_size)
    gen_steps = _chunk(_prefix_before_final(generated), step_size)
    correct = 0
    for oracle_step, gen_step in zip(oracle_steps, gen_steps):
        if oracle_step != gen_step:
            break
        correct += 1
    return correct / max(1, len(oracle_steps))


def step_level_metrics(task: FormalTask, example: Example, generated: Sequence[str]) -> Dict[str, float]:
    """Diagnostic step-level metrics for evaluation, independent of training reward_mode."""
    result = compute_dense_reward(task, example, generated, partial_credit=True)
    return {
        "step_alignment_accuracy": result.step_alignment_accuracy,
        "mean_step_reward": result.mean_step_reward,
        "num_correct_steps": float(result.num_correct_steps),
        "num_generated_steps": float(result.num_generated_steps),
        "num_oracle_steps": float(result.num_oracle_steps),
        "prefix_accuracy": prefix_accuracy(example, generated, task.step_size()),
    }
