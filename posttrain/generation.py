"""Text generation backends.  PROVIDED -- do not modify.

* hf_generate:    uses the in-memory policy (used for GRPO rollouts; no weight syncing).
* vllm_generate:  loads a saved checkpoint into vLLM (fast; used for evaluation,
                  preference-pair sampling and test-time scaling).  Optional: every
                  script falls back to hf_generate when vLLM is not installed.
"""

from __future__ import annotations

import gc
from typing import Optional

import torch

from posttrain.countdown import STOP_STRING
from posttrain.utils import autocast


def truncate_at_stop(text: str, stop: str = STOP_STRING) -> str:
    idx = text.find(stop)
    return text if idx < 0 else text[: idx + len(stop)]


def vllm_available() -> bool:
    try:
        import vllm  # noqa: F401

        return True
    except ImportError:
        return False


@torch.no_grad()
def hf_generate(
    model,
    tokenizer,
    prompts: list[str],
    n: int = 1,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
    batch_size: int = 32,
) -> list[list[str]]:
    """Returns completions[i][j] = j-th sample for prompts[i] (truncated after </answer>)."""
    was_training = model.training
    use_cache = model.config.use_cache
    model.eval()
    model.config.use_cache = True
    tokenizer.padding_side = "left"
    results: list[list[str]] = []
    try:
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            kwargs = dict(
                max_new_tokens=max_new_tokens,
                num_return_sequences=n,
                pad_token_id=tokenizer.pad_token_id,
            )
            if temperature > 0:
                kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
            else:
                kwargs.update(do_sample=False)
            with autocast():
                try:
                    out = model.generate(**enc, **kwargs, stop_strings=[STOP_STRING], tokenizer=tokenizer)
                except (TypeError, ValueError):  # older transformers without stop_strings
                    out = model.generate(**enc, **kwargs)
            texts = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
            texts = [truncate_at_stop(x) for x in texts]
            results.extend(texts[i * n : (i + 1) * n] for i in range(len(batch)))
    finally:
        tokenizer.padding_side = "right"
        model.config.use_cache = use_cache
        model.train(was_training)
    return results


def _mean_logprob(output) -> Optional[float]:
    n_tokens = len(output.token_ids)
    if n_tokens == 0:
        return None
    if getattr(output, "cumulative_logprob", None) is not None:
        return output.cumulative_logprob / n_tokens
    if output.logprobs:
        total = sum(step[tok].logprob for step, tok in zip(output.logprobs, output.token_ids))
        return total / n_tokens
    return None


def vllm_generate(
    model_path: str,
    prompts: list[str],
    n: int = 1,
    max_new_tokens: int = 384,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int = 0,
    return_confidence: bool = False,
    gpu_memory_utilization: float = 0.85,
):
    """Returns completions[i][j]; with return_confidence also mean token log-probs[i][j]."""
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_path, dtype="bfloat16", seed=seed, gpu_memory_utilization=gpu_memory_utilization)
    params = SamplingParams(
        n=n,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        stop=[STOP_STRING],
        include_stop_str_in_output=True,
        seed=seed,
        logprobs=1 if return_confidence else None,
    )
    outputs = llm.generate(prompts, params)
    texts = [[o.text for o in r.outputs] for r in outputs]
    conf = [[_mean_logprob(o) for o in r.outputs] for r in outputs] if return_confidence else None
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (texts, conf) if return_confidence else texts


@torch.no_grad()
def hf_mean_logprob(model, tokenizer, prompts: list[str], completions: list[str], batch_size: int = 16) -> list[float]:
    """Mean token log-prob of each completion under the model (teacher forcing).

    Uses YOUR tokenize_prompt_and_output / get_response_log_probs / masked_mean.
    """
    from posttrain.common import get_response_log_probs, masked_mean, tokenize_prompt_and_output

    model.eval()
    scores: list[float] = []
    for start in range(0, len(prompts), batch_size):
        p, c = prompts[start : start + batch_size], completions[start : start + batch_size]
        enc = {k: v.to(model.device) for k, v in tokenize_prompt_and_output(p, c, tokenizer).items()}
        with autocast():
            lp = get_response_log_probs(model, enc["input_ids"], enc["labels"])["log_probs"].float()
        mask = enc["response_mask"]
        for b in range(lp.shape[0]):
            scores.append(masked_mean(lp[b], mask[b]).item() if mask[b].any() else float("nan"))
    return scores


def generate(
    model_path: str,
    prompts: list[str],
    n: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float = 1.0,
    seed: int = 0,
    backend: str = "auto",
    return_confidence: bool = False,
    batch_size: int = 32,
):
    """Backend-agnostic generation from a saved checkpoint directory."""
    if backend == "auto":
        backend = "vllm" if vllm_available() and torch.cuda.is_available() else "hf"
    print(f"[generate] backend={backend} prompts={len(prompts)} n={n} T={temperature}")
    if backend == "vllm":
        return vllm_generate(
            model_path, prompts, n, max_new_tokens, temperature, top_p, seed, return_confidence=return_confidence
        )

    from posttrain.utils import load_model_and_tokenizer, set_seed

    set_seed(seed)
    model, tokenizer = load_model_and_tokenizer(model_path, dtype=torch.bfloat16)
    texts = hf_generate(model, tokenizer, prompts, n, max_new_tokens, temperature, top_p, batch_size)
    if not return_confidence:
        return texts
    flat_p = [p for p, outs in zip(prompts, texts) for _ in outs]
    flat_c = [c for outs in texts for c in outs]
    flat_s = hf_mean_logprob(model, tokenizer, flat_p, flat_c)
    conf, k = [], 0
    for outs in texts:
        conf.append(flat_s[k : k + len(outs)])
        k += len(outs)
    return texts, conf
