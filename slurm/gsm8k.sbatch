#!/bin/bash
#SBATCH --job-name=hw-gsm8k
#SBATCH --time=00:30:00
#SBATCH --account=cse595f26_class         # course Slurm account
#SBATCH --partition=spgpu                  # A40 48GB nodes
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
set -eo pipefail   # (no -u: Lmod and venv scripts reference unset variables)
source slurm/env.sh
# Problem 7: model A = Qwen2.5-0.5B-Instruct, model B = Qwen2.5-1.5B-Instruct (first 100 GSM8K test problems).
# Run with PROMPTS=standard first; once improved_prompt_template is written, PROMPTS=improved re-runs only (b).
A=Qwen/Qwen2.5-0.5B-Instruct
B=Qwen/Qwen2.5-1.5B-Instruct
PROMPTS="${PROMPTS:-standard improved}"
for P in ${PROMPTS}; do
  if [ "${P}" = standard ]; then
    python scripts/eval_gsm8k.py --model "${A}" --prompt standard --out results/gsm8k/A_standard
    python scripts/eval_gsm8k.py --model "${B}" --prompt standard --out results/gsm8k/B_standard
  else
    python scripts/eval_gsm8k.py --model "${A}" --prompt improved --out results/gsm8k/A_improved
  fi
done
