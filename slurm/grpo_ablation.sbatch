#!/bin/bash
#SBATCH --job-name=hw-grpo-abl
#SBATCH --time=01:15:00
#SBATCH --account=cse595f26_class         # course Slurm account
#SBATCH --partition=spgpu                  # A40 48GB nodes
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
set -eo pipefail   # (no -u: Lmod and venv scripts reference unset variables)
source slurm/env.sh
# Start from your own SFT model, or from the staff one:  USE_STAFF_SFT=1 sbatch slurm/<this file>
SFT_CKPT=results/sft/final
if [ "${USE_STAFF_SFT:-0}" = 1 ]; then SFT_CKPT="${STAFF_SFT}"; fi
echo "starting from SFT checkpoint: ${SFT_CKPT}"
# Write-up Q3 ablation (40 steps).  Pass the flags through ABLATION, for example
#   ABLATION="--baseline loo --no-std-normalization" sbatch slurm/grpo_ablation.sbatch
if [ -z "${ABLATION:-}" ]; then echo "set ABLATION, e.g. ABLATION=\"--baseline loo --no-std-normalization\""; exit 1; fi
echo "ablation flags: ${ABLATION}"
python scripts/run_grpo.py --policy "${SFT_CKPT}" --out results/grpo_ablation --steps 40 ${ABLATION}
