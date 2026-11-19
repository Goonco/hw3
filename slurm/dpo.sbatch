#!/bin/bash
#SBATCH --job-name=hw-dpo
#SBATCH --time=01:00:00
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
# data/pref_pairs.jsonl is shipped (built from the staff SFT model), so pair construction is skipped.
# Delete or rename it to build your own pairs from ${SFT_CKPT} (+~15 min): greedy-decode 7000 training
# prompts, then pair each wrong greedy completion (rejected) with a correct sample (chosen).
if [ ! -f data/pref_pairs.jsonl ]; then
  python scripts/evaluate.py --model "${SFT_CKPT}" --split data/countdown_train.jsonl --limit 7000 --out results/sft_train_greedy
  python scripts/make_pref_pairs.py --model "${SFT_CKPT}" --greedy-predictions results/sft_train_greedy/predictions.jsonl --out data/pref_pairs.jsonl
fi
python scripts/run_dpo.py --policy "${SFT_CKPT}" --pairs data/pref_pairs.jsonl --out results/dpo
python scripts/evaluate.py --model results/dpo/final --out results/dpo
