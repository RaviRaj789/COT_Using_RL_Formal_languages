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
