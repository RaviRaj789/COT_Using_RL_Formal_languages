For your ML/RL research workflow on the CWRU HPC, these are the commands you'll use 95% of the time.

# 1. Login to HPC

```bash
ssh rxk789@hpc7.case.edu
```

Check where you are:

```bash
hostname
```

You should see something like:

```text
hpc7
```

(login node)

---

# 2. Check group access

```bash
groups
```

Expected:

```text
sxr358
```

Check Slurm account:

```bash
sacctmgr show associations user=rxk789 format=User,Account,Partition,QOS
```

Expected:

```text
rxk789  sxr358  normal
```

---

# 3. Check available resources

Partitions:

```bash
sinfo
```

Nodes:

```bash
sinfo -N
```

Partitions details:

```bash
scontrol show partition
```

---

# 4. See GPU resources

Request a GPU interactively:

```bash
salloc --account=sxr358 --partition=gpu --gres=gpu:1 --nodes=1 --cpus-per-task=4 --mem=32G --time=02:00:00 --exclude=gput052

salloc --account=sxr358 --partition=gpu --gres=gpu:1 --nodes=1 --cpus-per-task=4 --mem=32G --time=02:00:00   --exclude=gput052,gput065
```

Enter allocated node:

```bash
srun --pty bash
```

Check node:

```bash
hostname
```

Check GPU:

```bash
nvidia-smi
```

Check GPU type:

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv
```
```bash
module purge
module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0
module load tqdm/4.64.0-GCCcore-11.3.0
```
---

# 5. Exit GPU allocation

Leave shell:

```bash
exit
```

Cancel allocation:

```bash
squeue -u $USER
scancel JOB_ID
```

Find job id:

```bash
squeue -u rxk789
```

---

# 6. Monitor jobs

Your jobs:

```bash
squeue -u rxk789
```

All jobs:

```bash
squeue
```

Job details:

```bash
scontrol show job JOB_ID
```

Job accounting:

```bash
sacct -j JOB_ID
```

---

# 7. Submit batch jobs

Create:

```bash
nano train.slurm
```

Example:

```bash
#!/bin/bash
#SBATCH --account=sxr358
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=train.out

python train.py
```

Submit:

```bash
sbatch train.slurm
```

---

# 8. File operations

Current directory:

```bash
pwd
```

List files:

```bash
ls -lh
```

Disk usage:

```bash
du -sh .
```

Storage quota:

```bash
quota -s
```

Find large files:

```bash
du -sh * | sort -h
```

---

# 9. Conda environments

Load module:

```bash
module load Miniconda3/23.10.0-1
```

Create env:

```bash
conda create -n rl python=3.11
```

Activate:

```bash
conda activate rl
```

Deactivate:

```bash
conda deactivate
```

List envs:

```bash
conda env list
```

---

# 10. Install packages

```bash
pip install torch torchvision torchaudio
```

Save packages:

```bash
pip freeze > requirements.txt
```

---

# 11. Transfer files

Laptop → HPC:

```bash
scp file.py rxk789@hpc7.case.edu:~
```

Folder:

```bash
scp -r project rxk789@hpc7.case.edu:~
```

HPC → laptop:

```bash
scp rxk789@hpc7.case.edu:~/results.csv .
```

---

# 12. Git

Clone:

```bash
git clone REPO_URL
```

Pull:

```bash
git pull
```

Push:

```bash
git add .
git commit -m "update"
git push
```

---

# 13. Useful monitoring

CPU usage:

```bash
top
```

Modern version:

```bash
htop
```

GPU usage:

```bash
watch -n 1 nvidia-smi
```

Memory:

```bash
free -h
```

---

# 14. VS Code Remote SSH

On laptop:

```bash
code .
```

Install:

* Remote SSH extension

Connect:

```text
rxk789@hpc7.case.edu
```

Then run jobs through Slurm from the VS Code terminal.

---

# 15. Research workflow for your PPO/GRPO project

Login:

```bash
ssh rxk789@hpc7.case.edu
```

Go to project:

```bash
cd formal_rl_length_generalization
```

Request GPU:

```bash
salloc --account=sxr358 --partition=gpu --gres=gpu:1 --time=08:00:00
```

Enter node:

```bash
srun --pty bash
```

Activate env:

```bash
conda activate rl
```

Train:

```bash
python -m formal_rl_length_generalization.train --config configs/parity_grpo_process_terminal.yaml
```

Monitor GPU:

```bash
watch -n 1 nvidia-smi
```

Evaluate:

```bash
python -m formal_rl_length_generalization.eval --checkpoint runs/parity_grpo_process_terminal/checkpoint.pt
```

Release resources:

```bash
exit
```

and if needed:

```bash
scancel JOB_ID
```

These commands are essentially the complete day-to-day toolkit you'll need for running your formal-language length-generalization experiments on the CWRU HPC cluster.




Use Slurm. Do **not** train directly on `hpc7`; that is the login node.

## Interactive GPU training

```bash
salloc --account=sxr358 --partition=gpu --gres=gpu:1 --nodes=1 --time=02:00:00
```

Then enter the allocated GPU node:

```bash
srun --pty bash
```

Check you are on the GPU node:

```bash
hostname
```

Check GPU:

```bash
nvidia-smi
```

Run training:

```bash
cd ~/formal_rl_length_generalization && conda activate rl && python -m formal_rl_length_generalization.train --config configs/parity_grpo_process_terminal.yaml
```

Release GPU:

```bash
exit
```

Then check/cancel if still running:

```bash
squeue -u rxk789
scancel JOB_ID
```

## One-line GPU allocation + shell

```bash
salloc --account=sxr358 --partition=gpu --gres=gpu:1 --nodes=1 --time=02:00:00 && srun --pty bash
```

## Batch GPU training

Create:

```bash
nano train_gpu.slurm
```

Paste:

```bash
#!/bin/bash
#SBATCH --account=sxr358
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --job-name=grpo_parity
#SBATCH --output=runs/grpo_parity_%j.out

cd ~/formal_rl_length_generalization
source ~/.bashrc
conda activate rl

python -m formal_rl_length_generalization.train --config configs/parity_grpo_process_terminal.yaml
```

Submit:

```bash
sbatch train_gpu.slurm
```

Monitor:

```bash
squeue -u rxk789
tail -f runs/grpo_parity_JOBID.out
```

Cancel:

```bash
scancel JOB_ID
```

## Most useful GPU commands

```bash
sinfo -p gpu
```

```bash
squeue -p gpu
```

```bash
nvidia-smi
```

```bash
watch -n 1 nvidia-smi
```

