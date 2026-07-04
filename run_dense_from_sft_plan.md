# Plan: dense-reward GRPO warm-started from the SFT checkpoint

Goal: test whether the dense per-step reward (`formal_rl_length_generalization/dense_reward.py`)
improves on the sequence-mode warm start (`runs/parity_grpo_from_sft_safe`, which holds
`train_1_40` terminal accuracy in the 0.80-0.95 range) instead of collapsing like the
earlier cold-start dense run (`runs/parity_grpo_dense_process_terminal_001`, stuck near
0% the whole run) or the unstable warm start (`runs/parity_grpo_from_sft`, degraded from
0.875 to 0.225 due to too-high lr/entropy_coef/temperature).

New config: `configs/parity_grpo_dense_from_sft.yaml` -- identical "safe" hyperparameters
to `configs/parity_grpo_from_sft.yaml` (`lr=5e-6, clip_eps=0.1, entropy_coef=0.0,
generation.temperature=0.3`, 500 steps, eval every 50), but with `reward_mode: dense`
turned on. Only the reward signal differs from the known-stable baseline.

---

## 1. Sanity-check the dense reward code first

```bash
pytest tests/test_dense_reward.py -s
```

## 2. Confirm the SFT checkpoint exists

```bash
ls runs/parity_sft/checkpoint.pt
```

## 3. Run dense-reward GRPO warm-started from SFT (local smoke test)

```bash
python -m formal_rl_length_generalization.train \
  --config configs/parity_grpo_dense_from_sft.yaml \
  --init-from-checkpoint runs/parity_sft/checkpoint.pt
```

Watch for:
- `entropy` starting low (near the SFT policy's entropy, not ~6.8) -- confirms the
  warm start actually loaded.
- `train_1_40` `terminal` staying in the 0.7-0.95 range across checkpoints instead of
  collapsing toward 0, matching `runs/parity_grpo_from_sft_safe`.
- `dense_process_reward` / `mean_step_reward` moving upward over training.

## 4. On the HPC (full run / longer steps if the local smoke test looks healthy)

```bash
ssh rxk789@hpc7.case.edu
cd ~/formal_rl_length_generalization
git pull
conda activate rl
salloc --account=sxr358 --partition=gpu --gres=gpu:1 --nodes=1 --cpus-per-task=4 --mem=32G --time=04:00:00
srun --pty bash
python -m formal_rl_length_generalization.train \
  --config configs/parity_grpo_dense_from_sft.yaml \
  --init-from-checkpoint runs/parity_sft/checkpoint.pt
```

## 5. Compare against the sequence-mode warm start

```bash
python -m formal_rl_length_generalization.eval \
  --checkpoint runs/parity_grpo_dense_from_sft/checkpoint.pt \
  --out runs/parity_grpo_dense_from_sft/eval.json

python -m formal_rl_length_generalization.eval \
  --checkpoint runs/parity_grpo_from_sft_safe/checkpoint.pt \
  --out runs/parity_grpo_from_sft_safe/eval.json
```

Diff `step_accuracy`, `mean_step_reward`, and `terminal`/`final_answer_accuracy` per
bucket between the two `eval.json` files, especially in the OOD buckets
(`ood_161_320`, `ood_321_500`) -- that's the comparison this experiment is for.
