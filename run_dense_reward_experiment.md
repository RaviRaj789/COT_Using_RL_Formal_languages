# Running the dense-reward experiment on the CWRU HPC

Step-by-step commands to test and train the new dense per-step process reward
(`formal_rl_length_generalization/dense_reward.py`,
`configs/parity_grpo_dense_process_terminal.yaml`) on the HPC.

---

## 1. Commit and push locally

```bash
git add formal_rl_length_generalization/dense_reward.py formal_rl_length_generalization/tasks.py formal_rl_length_generalization/rl.py formal_rl_length_generalization/train.py formal_rl_length_generalization/evaluate.py configs/parity_grpo_dense_process_terminal.yaml tests/ README.md
git commit -m "add dense per-step process reward mode"
git push
```

---

## 2. Log in and pull the changes

```bash
ssh rxk789@hpc7.case.edu
cd ~/formal_rl_length_generalization
git pull
```

---

## 3. Sanity-check the reward math (CPU, login node, fast)

```bash
conda activate rl
pytest tests/test_dense_reward.py -s
```

All tests must pass before spending GPU time — this checks partial credit,
step alignment, and terminal reward assignment that the RL update directly
optimizes against.

---

## 4. Request a GPU

```bash
salloc --account=sxr358 --partition=gpu --gres=gpu:1 --nodes=1 --cpus-per-task=4 --mem=32G --time=08:00:00
srun --pty bash
hostname        # confirm you're on the GPU node, not hpc7
nvidia-smi
conda activate rl
```

---

## 5. Interactive smoke test (short run)

```bash
cd ~/formal_rl_length_generalization
python -m formal_rl_length_generalization.train --config configs/parity_grpo_dense_process_terminal.yaml --steps 20
```

Watch stdout / `train_metrics.json` for `dense_process_reward` moving
sensibly before committing to the full run.

---

## 6. Full training run (batch job)

```bash
nano train_dense.slurm
```

```bash
#!/bin/bash
#SBATCH --account=sxr358
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --job-name=grpo_dense
#SBATCH --output=runs/grpo_dense_%j.out

cd ~/formal_rl_length_generalization
source ~/.bashrc
conda activate rl

python -m formal_rl_length_generalization.train --config configs/parity_grpo_dense_process_terminal.yaml
```

```bash
sbatch train_dense.slurm
squeue -u rxk789
tail -f runs/grpo_dense_JOBID.out
```

Optional: submit the sequence-mode baseline for comparison, same job file
but `--config configs/parity_grpo_process_terminal.yaml`.

---

## 7. Evaluate the checkpoint

Run dir defaults to `runs/parity_grpo_dense_process_terminal` (from
`run_name` in the config; a `_001` suffix is appended if that dir already
exists).

```bash
python -m formal_rl_length_generalization.eval \
  --checkpoint runs/parity_grpo_dense_process_terminal/checkpoint.pt \
  --out runs/parity_grpo_dense_process_terminal/eval.json
```

Reports `step_accuracy`, `mean_step_reward`, `prefix_accuracy`, and
`step_accuracy_by_length` for every length bucket (including OOD), so you
can diff this against the sequence-mode run's `eval.json`.

---

## 8. Release the GPU

```bash
exit
squeue -u rxk789
scancel JOB_ID   # only if something is still stuck/running
```

---

## 9. Pull results back to your laptop

```bash
scp -r rxk789@hpc7.case.edu:~/formal_rl_length_generalization/runs/parity_grpo_dense_process_terminal .
```
