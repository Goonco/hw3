#!/bin/bash
#SBATCH --job-name=hw-sft
#SBATCH --time=01:00:00
#SBATCH --account=cse595f26_class         # course Slurm account
#SBATCH --partition=spgpu                  # A40 48GB nodes
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
set -eo pipefail   # (no -u: Lmod and venv scripts reference unset variables)
source slurm/env.sh
# Base-model baseline for the write-up (only once).
[ -f results/base/metrics.json ] || python scripts/evaluate.py --model Qwen/Qwen2.5-0.5B --out results/base
python scripts/run_sft.py --out results/sft
python scripts/evaluate.py --model results/sft/final --out results/sft
