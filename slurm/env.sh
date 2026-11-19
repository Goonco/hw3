# Sourced by every job script (run sbatch from the top of your starter-code folder).
export VENV_DIR="${HOME}/venvs/posttrain"       # Python environment lives in your home directory
# Staff SFT checkpoint (fallback starting point for DPO/GRPO; see Problem 3 in hw3.pdf).
export STAFF_SFT="${PWD}/staff_sft"   # shipped with the starter code
export HF_HOME="${HOME}/.cache/huggingface"     # model downloads go to your home directory
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # avoids allocator fragmentation (peak memory 40 GB -> 24-31 GB)
export VLLM_USE_FLASHINFER_SAMPLER=0   # vLLM>=0.2x: FlashInfer sampler JIT-compiles and needs nvcc (absent on compute nodes)
mkdir -p "${HF_HOME}" logs
module load python/3.11.5   # the version the staff reference run used
source "${VENV_DIR}/bin/activate"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
