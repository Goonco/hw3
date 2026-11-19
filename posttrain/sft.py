"""Part 2: supervised fine-tuning (SFT).

Implement every function marked TODO.  Tests: tests/test_sft.py
"""

from __future__ import annotations

import torch

from posttrain.common import masked_normalize


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """One SFT micro-batch: compute the loss and call ``backward()`` on it.

    policy_log_probs: (B, T) per-token log-probs of the target tokens (requires grad)
    response_mask:    (B, T) True on response tokens

    Loss definition (use masked_normalize):
        per_example[b] = sum_t response_mask[b,t] * log_probs[b,t] / normalize_constant
        loss = -mean_b(per_example) / gradient_accumulation_steps

    Call loss.backward() inside this function, then return
        (loss, {"num_response_tokens": <python int>})
    where ``loss`` is the scaled scalar that you called backward on.
    """
    per_example = masked_normalize(policy_log_probs, response_mask, normalize_constant, dim=-1)
    loss = -per_example.mean() / gradient_accumulation_steps
    loss.backward()
    return loss, {"num_response_tokens": response_mask.count_nonzero().item()}
