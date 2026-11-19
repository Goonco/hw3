"""Part 1: shared utilities used by SFT, DPO and GRPO.

Implement every function marked TODO.  Tests: tests/test_common.py
"""

from __future__ import annotations

from typing import Optional

import torch


def tokenize_prompt_and_output(prompt_strs: list[str], output_strs: list[str], tokenizer) -> dict[str, torch.Tensor]:
    """Tokenize prompt/output pairs into a right-padded batch for teacher forcing.

    Tokenize each prompt and each output SEPARATELY with
    ``tokenizer.encode(text, add_special_tokens=False)`` and concatenate them.
    Let L be the length of the longest concatenated sequence.  Right-pad every
    sequence to length L with ``tokenizer.pad_token_id``, giving a (B, L) tensor
    ``full``.  Return

        input_ids      = full[:, :-1]                       LongTensor (B, L-1)
        labels         = full[:, 1:]                        LongTensor (B, L-1)
        response_mask  = mask over ``labels`` that is True where the label token
                         belongs to the OUTPUT (not the prompt, not padding).
                         BoolTensor (B, L-1)
    """
    if len(prompt_strs) != len(output_strs):
        raise ValueError("Expected one output per prompt")

    prompts = [tokenizer.encode(text, add_special_tokens=False) for text in prompt_strs]
    outputs = [tokenizer.encode(text, add_special_tokens=False) for text in output_strs]
    sequences = [prompt + output for prompt, output in zip(prompts, outputs)]
    width = max(map(len, sequences), default=0)
    full = torch.full((len(sequences), width), tokenizer.pad_token_id, dtype=torch.long)
    for row, sequence in enumerate(sequences):
        full[row, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)

    positions = torch.arange(width)[1:]
    starts = torch.tensor([len(prompt) for prompt in prompts], dtype=torch.long)[:, None]
    ends = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)[:, None]
    response_mask = (positions >= starts) & (positions < ends)
    return {"input_ids": full[:, :-1], "labels": full[:, 1:], "response_mask": response_mask}


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Entropy (in nats) of the softmax distribution over the last dimension.

    logits: (..., V)  ->  returns (...,)
    Must be numerically stable: logits with magnitude ~1e3 must not produce nan/inf.
    """
    log_probs = logits.log_softmax(dim=-1)
    return -(log_probs.exp() * log_probs).sum(dim=-1)


def get_response_log_probs(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """Per-token log p(labels[b, t] | input_ids[b, :t+1]) under ``model``.

    Call the model as ``model(input_ids).logits`` (shape (B, T, V)).
    Returns {"log_probs": (B, T)} and, if requested, {"token_entropy": (B, T)}
    (the entropy of the next-token distribution at every position; use
    compute_entropy).  Compute log-probabilities in float32.
    """
    logits = model(input_ids).logits.float()
    label_log_probs = logits.log_softmax(dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    result = {"log_probs": label_log_probs}
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)
    return result


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    """Mean of ``tensor`` over the elements where ``mask`` is nonzero.

    dim=None averages over everything (returns a scalar); otherwise averages along
    ``dim``.  ``mask`` has the same shape as ``tensor`` and may be bool or numeric.
    You may assume every slice being averaged has at least one unmasked element.
    """
    active = mask.bool()
    return tensor.masked_fill(~active, 0).sum(dim=dim) / active.sum(dim=dim)


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: Optional[int] = None,
) -> torch.Tensor:
    """Sum of ``tensor`` over unmasked elements, divided by ``normalize_constant``.

    dim=None sums over everything (returns a scalar); otherwise sums along ``dim``.
    """
    return tensor.masked_fill(~mask.bool(), 0).sum(dim=dim) / normalize_constant
