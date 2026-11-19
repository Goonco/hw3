#!/bin/bash
# One-time setup, run on a LOGIN node:   bash slurm/setup_env.sh
set -eo pipefail   # (no -u: Lmod and venv scripts reference unset variables)
mkdir -p logs        # Slurm writes job output to logs/ (see #SBATCH --output) but does not create it
source "$(dirname "$0")/env.sh" 2>/dev/null || true   # sets VENV_DIR (activation fails the first time; fine)
python -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt -c requirements-lock.txt
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"
