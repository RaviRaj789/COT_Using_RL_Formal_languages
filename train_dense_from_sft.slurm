#!/bin/bash
#SBATCH --account=sxr358
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=grpo_dense_from_sft
#SBATCH --output=runs/grpo_dense_from_sft_%j.out

cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
conda activate rl

python -m formal_rl_length_generalization.train \
  --config configs/parity_grpo_dense_from_sft.yaml \
  --init-from-checkpoint runs/parity_sft/checkpoint.pt
