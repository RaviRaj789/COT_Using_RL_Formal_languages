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
