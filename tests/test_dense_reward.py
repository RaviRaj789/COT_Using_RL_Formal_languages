"""Sanity checks for dense process rewards on the parity task.

Each test builds a hand-constructed (example, generated-trace) pair covering one
of the scenarios called out for length-generalization RL: fully correct traces,
correct-final/wrong-reasoning, partially correct reasoning, correct-reasoning/
wrong-final, and missing/extra intermediate steps. Run with:

    pytest tests/test_dense_reward.py -s
"""

from formal_rl_length_generalization.dense_reward import compute_dense_reward, step_level_metrics
from formal_rl_length_generalization.tasks import Example, ParityTask

task = ParityTask()


def make_example() -> Example:
    xs = ["1", "1", "0", "1", "0"]
    trace = ["P1", "P0", "P0", "P1", "P1"]
    return Example("parity", xs, trace, "P1")


def test_fully_correct_trace():
    ex = make_example()
    generated = ["P1", "P0", "P0", "P1", "P1", "FINAL", "P1"]
    result = compute_dense_reward(task, ex, generated)
    print("fully_correct:", result.dense_process_reward, result.terminal_reward, result.total_reward)

    assert result.terminal_reward == 1.0
    assert result.dense_process_reward == 1.0
    assert result.sequence_process_reward == 1.0
    assert result.num_correct_steps == 5
    assert result.num_generated_steps == 5
    assert result.num_oracle_steps == 5
    assert result.step_alignment_accuracy == 1.0
    assert result.total_reward == 2.0


def test_reasoning_correct_final_wrong():
    ex = make_example()
    generated = ["P1", "P0", "P0", "P1", "P1", "FINAL", "P0"]
    result = compute_dense_reward(task, ex, generated)
    print("reasoning_correct_final_wrong:", result.dense_process_reward, result.terminal_reward)

    assert result.terminal_reward == 0.0
    assert result.dense_process_reward == 1.0
    assert result.total_reward == 1.0


def test_final_correct_reasoning_wrong():
    ex = make_example()
    generated = ["P0", "P1", "P1", "P0", "P0", "FINAL", "P1"]
    result = compute_dense_reward(task, ex, generated)
    print("final_correct_reasoning_wrong:", result.dense_process_reward, result.terminal_reward)

    assert result.terminal_reward == 1.0
    assert result.dense_process_reward < 1.0
    assert result.dense_process_reward <= 0.7


def test_reasoning_partially_correct():
    ex = make_example()
    correct_prefix = ["P1", "P0", "P1", "P0", "P0", "FINAL", "P1"]
    fully_wrong = ["P0", "P1", "P1", "P0", "P0", "FINAL", "P1"]

    partial = compute_dense_reward(task, ex, correct_prefix)
    wrong = compute_dense_reward(task, ex, fully_wrong)
    print("reasoning_partially_correct:", partial.dense_process_reward, partial.terminal_reward)

    assert partial.terminal_reward == 1.0
    assert 0.0 < partial.dense_process_reward < 1.0
    assert partial.dense_process_reward > wrong.dense_process_reward


def test_missing_intermediate_step():
    ex = make_example()
    # oracle trace has 5 steps ["P1","P0","P0","P1","P1"]; drop the third one ("P0" at index 2)
    generated = ["P1", "P0", "P1", "P1", "FINAL", "P1"]
    result = compute_dense_reward(task, ex, generated)
    print(
        "missing_step: dense=%.4f terminal=%.2f mean_step=%.4f correct=%d align_acc=%.2f total=%.4f"
        % (
            result.dense_process_reward,
            result.terminal_reward,
            result.mean_step_reward,
            result.num_correct_steps,
            result.step_alignment_accuracy,
            result.total_reward,
        )
    )

    assert result.terminal_reward == 1.0
    assert result.num_generated_steps == 4
    assert result.num_oracle_steps == 5
    # a single dropped step costs that step's credit (0) plus a symbol-alignment
    # penalty on steps after it, but the two exactly-correct steps before the drop,
    # and the recoverable prev-state/transition credit after it, are not wiped out
    assert result.dense_process_reward == 0.55
    assert result.mean_step_reward == 0.6875
    assert result.num_correct_steps == 1
    assert result.step_alignment_accuracy == 0.8
    assert result.total_reward == 1.55


def test_extra_intermediate_step():
    ex = make_example()
    # oracle trace ["P1","P0","P0","P1","P1"] with an extra "P1" spliced in after step 0
    generated = ["P1", "P1", "P0", "P0", "P1", "P1", "FINAL", "P1"]
    result = compute_dense_reward(task, ex, generated)
    print(
        "extra_step: dense=%.4f terminal=%.2f align_acc=%.2f total=%.4f"
        % (result.dense_process_reward, result.terminal_reward, result.step_alignment_accuracy, result.total_reward)
    )

    assert result.terminal_reward == 1.0
    assert result.num_generated_steps == 6
    assert result.num_oracle_steps == 5
    # the oracle trace is a subsequence of the generated one, so alignment should
    # still recover most of the credit despite the extra step
    assert result.dense_process_reward > 0.4


def test_sequence_mode_matches_legacy_task_reward():
    ex = make_example()
    generated = ["P1", "P0", "P1", "P0", "P0", "FINAL", "P1"]
    legacy = task.reward(ex, generated)
    dense = compute_dense_reward(task, ex, generated)

    assert dense.sequence_process_reward == legacy.process
    assert dense.terminal_reward == legacy.terminal


def test_trailing_eos_token_is_stripped_and_padded():
    ex = make_example()
    generated = ["P1", "P0", "P0", "P1", "P1", "FINAL", "P1", "<eos>"]
    result = compute_dense_reward(task, ex, generated)

    assert len(result.token_rewards) == len(generated)
    assert result.terminal_reward == 1.0
    assert result.dense_process_reward == 1.0
    assert result.token_rewards[-1] == 0.0


def test_partial_credit_disabled_is_binary_per_step():
    ex = make_example()
    generated = ["P1", "P0", "P1", "P0", "P0", "FINAL", "P1"]
    result = compute_dense_reward(task, ex, generated, partial_credit=False)

    for step in result.step_scores:
        assert step.total in (0.0, 1.0)


def test_step_level_metrics_eval_helper():
    ex = make_example()
    generated = ["P1", "P0", "P0", "P1", "P1", "FINAL", "P1"]
    metrics = step_level_metrics(task, ex, generated)

    assert metrics["prefix_accuracy"] == 1.0
    assert metrics["step_alignment_accuracy"] == 1.0
    assert metrics["num_oracle_steps"] == 5.0


def test_prefix_accuracy_stops_at_first_mismatch():
    from formal_rl_length_generalization.dense_reward import prefix_accuracy

    ex = make_example()
    generated = ["P1", "P0", "P1", "P1", "P1", "FINAL", "P1"]  # first 2 steps correct, then diverges
    assert prefix_accuracy(ex, generated, task.step_size()) == 2 / 5
