"""Small shared helpers.  PROVIDED -- do not modify."""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def read_jsonl(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_json(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast():
    """bf16 autocast on GPU (master weights stay fp32); a no-op on CPU."""
    if torch.cuda.is_available():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


def load_model_and_tokenizer(path: str, dtype=torch.float32, gradient_checkpointing: bool = False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype).to(device())
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    return model, tokenizer


def save_model(model, tokenizer, out_dir) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    was_cache = getattr(model.config, "use_cache", True)
    model.config.use_cache = True
    # Save a bf16 COPY of the weights (small checkpoints, vLLM-friendly) without
    # touching the fp32 weights that training continues from.
    state = {k: v.detach().to(torch.bfloat16).cpu() for k, v in model.state_dict().items()}
    model.config.torch_dtype = torch.bfloat16
    model.save_pretrained(out_dir, state_dict=state, safe_serialization=True)
    model.config.use_cache = was_cache
    tokenizer.save_pretrained(out_dir)
    print(f"saved checkpoint to {out_dir}")


def warmup_cosine(optimizer, warmup_steps: int, total_steps: int, min_ratio: float = 0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class JsonlLogger:
    """Appends one JSON object per line and echoes a compact version to stdout."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "a")
        self.t0 = time.time()

    def log(self, **kv):
        kv["elapsed_s"] = round(time.time() - self.t0, 1)
        self.f.write(json.dumps(kv) + "\n")
        self.f.flush()
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in kv.items()), flush=True)


def env_info() -> dict:
    info = {"torch": torch.__version__, "cuda": torch.cuda.is_available()}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    info["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
    return info
