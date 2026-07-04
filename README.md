# formal_rl_length_generalization

PyTorch codebase for studying whether PPO and GRPO with oracle process rewards improve length generalization on formal-language tasks across the Chomsky hierarchy.

## Install

```bash
pip install -r requirements.txt
```

## Quick smoke run

```bash
python -m formal_rl_length_generalization.train --config configs/parity_sft.yaml --steps 50 --batch-size 16
python -m formal_rl_length_generalization.eval --checkpoint runs/parity_sft/checkpoint.pt
```

### Warm-start RL from an existing SFT checkpoint

Use a new run name in your config so the RL checkpoint is saved into a separate folder, while the original SFT checkpoint stays untouched.

```bash
python -m formal_rl_length_generalization.train \
  --config configs/parity_ppo_process_terminal.yaml \
  --steps 1000 \
  --init-from-checkpoint runs/parity_sft/checkpoint.pt
```

If you want the RL run to live in its own folder, set a different `run_name` in `configs/parity_ppo_process_terminal.yaml` before starting.

## Algorithms

- `sft`: supervised chain-of-thought baseline
- `ppo_terminal`: PPO with terminal reward only
- `ppo_process`: PPO with oracle process reward only
- `ppo_process_terminal`: PPO with oracle process plus terminal reward
- `grpo_terminal`: GRPO with terminal reward only
- `grpo_process`: GRPO with oracle process reward only
- `grpo_process_terminal`: GRPO with oracle process plus terminal reward

Training lengths are restricted to `1..40`. OOD evaluation lengths are `41..80`, `81..160`, `161..320`, and `321..500`.

## Tasks

- parity
- modular counting with `k=3` or `k=5`
- `a*b*`
- `a^n b^n`

The RL algorithms use oracle task states only for rewards. The policy sees text prompts and generates trace tokens plus final accept/reject or final state tokens.

## Reward modes

Every `*_process*` algorithm reads a `reward_mode` from `algorithm:` (or per curriculum stage) in the config:

- `reward_mode: sequence` (default) — unchanged original behavior. The whole generated trace is compared positionally against the oracle trace, reduced to one process-match scalar, combined with the terminal reward, and broadcast as the same reward value onto every generated token.
- `reward_mode: dense` — each reasoning step is aligned against the oracle trace (robust to missing/extra/reordered steps) and scored with 0.25-weighted partial credit for symbol position, previous-state consistency, transition correctness, and final state/value match. The resulting per-step rewards are assigned only to the generated tokens for that step, and the terminal reward is assigned to the `FINAL <answer>` tokens, giving PPO/GRPO a true per-token reward signal instead of one broadcast scalar. See `formal_rl_length_generalization/dense_reward.py`.

Other reward knobs (also settable per curriculum stage): `process_weight`, `terminal_weight`, `dense_step_weight`, `partial_credit` (set `false` for binary per-step credit instead of the 0.25-weighted components), `normalize_dense_reward` (divide the summed step rewards by the oracle step count before weighting).

```bash
# sequence-mode (default) GRPO training
python -m formal_rl_length_generalization.train --config configs/parity_grpo_from_sft.yaml --steps 300

# dense-mode GRPO training
python -m formal_rl_length_generalization.train --config configs/parity_grpo_dense_process_terminal.yaml --steps 300
```

`eval.py` / `evaluate_buckets` report step-level diagnostics (`step_accuracy`, `mean_step_reward`, `prefix_accuracy`, `step_accuracy_by_length`) for every bucket regardless of the reward mode a checkpoint was trained with, so `sequence`- and `dense`-trained runs can be compared on the same footing, including in the OOD length buckets.
