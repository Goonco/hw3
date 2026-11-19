"""Part 4: RL with verifiable rewards (GRPO / RLOO / Dr. GRPO variants).

Implement every function marked TODO.  Tests: tests/test_grpo.py
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch


def compute_group_advantages(
    raw_rewards: Sequence[float] | torch.Tensor,
    group_size: int,
    baseline: str = "mean",
    normalize_by_std: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Turn raw rewards into per-response advantages, one group per prompt.

    raw_rewards has length N = num_prompts * group_size and is ordered so that
    responses [g*group_size : (g+1)*group_size] belong to prompt g.

    For response i in a group with rewards r_1..r_G:
        baseline == "mean": b_i = mean_j r_j                     (GRPO)
        baseline == "loo":  b_i = (sum_{j != i} r_j) / (G - 1)   (RLOO, leave-one-out)
        A_i = r_i - b_i
    If normalize_by_std: A_i = A_i / (std_G + eps), where std_G is the POPULATION
    standard deviation of the group's raw rewards (divide by G, not G - 1).

    Returns a float32 tensor of shape (N,).
    """
    rewards = torch.as_tensor(raw_rewards, dtype=torch.float32).reshape(-1, group_size)
    shifted = rewards - rewards[:, :1]
    advantages = shifted - shifted.mean(dim=1, keepdim=True)
    if baseline == "loo":
        advantages = advantages * (group_size / max(group_size - 1, 1))
    elif baseline != "mean":
        raise ValueError(f"Unknown reward baseline: {baseline}")

    if normalize_by_std:
        std = shifted.std(dim=1, unbiased=False, keepdim=True)
        advantages = advantages / (std + eps)
    return advantages.flatten()


def per_token_policy_loss(
    policy_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: Optional[torch.Tensor] = None,
    loss_type: str = "grpo_clip",
    clip_eps: float = 0.2,
) -> tuple[torch.Tensor, dict]:
    """Per-token policy-gradient loss (to be MINIMIZED).

    policy_log_probs: (B, T); advantages: (B,) broadcast over tokens; old_log_probs: (B, T)

        loss_type == "pg":         loss[b,t] = -A[b] * logp[b,t]         (old_log_probs ignored)
        loss_type == "grpo_clip":  ratio = exp(logp - old_logp)
                                   loss[b,t] = -min(ratio * A[b], clip(ratio, 1-eps, 1+eps) * A[b])

    Return (loss (B, T), {"clipped": BoolTensor (B, T)}) where "clipped" is True exactly
    where the clipped term is STRICTLY smaller than the unclipped term (always False for "pg").
    """
    advantages = advantages[:, None]
    if loss_type == "pg":
        return -advantages * policy_log_probs, {
            "clipped": torch.zeros_like(policy_log_probs, dtype=torch.bool),
        }
    if loss_type != "grpo_clip":
        raise ValueError(f"Unknown policy loss: {loss_type}")
    if old_log_probs is None:
        raise ValueError("GRPO clipping requires old log-probabilities")

    ratio = (policy_log_probs - old_log_probs).exp()
    objective = ratio * advantages
    clipped_objective = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages
    return -torch.minimum(objective, clipped_objective), {
        "clipped": clipped_objective < objective,
    }


def aggregate_loss(
    per_token_loss: torch.Tensor,
    response_mask: torch.Tensor,
    aggregation: str = "seq_mean",
    normalize_constant: Optional[float] = None,
) -> torch.Tensor:
    """Reduce a (B, T) per-token loss to a scalar using only response tokens.

        "seq_mean":   average over each sequence's response tokens, then over the batch
                      (original GRPO; a sequence with zero response tokens contributes 0)
        "token_mean": sum over all response tokens / total number of response tokens
                      (DAPO-style)
        "constant":   sum over all response tokens / (B * normalize_constant)
                      (Dr. GRPO-style; normalize_constant is e.g. the max generation length)
    """
    mask = response_mask.bool()
    losses = per_token_loss.masked_fill(~mask, 0.0)
    if aggregation == "seq_mean":
        lengths = mask.sum(dim=1).clamp_min(1)
        return (losses.sum(dim=1) / lengths).mean()
    if aggregation == "token_mean":
        return losses.sum() / mask.sum().clamp_min(1)
    if aggregation == "constant":
        if normalize_constant is None or normalize_constant <= 0:
            raise ValueError("Constant aggregation requires a positive normalize_constant")
        return losses.sum() / (losses.shape[0] * normalize_constant)
    raise ValueError(f"Unknown aggregation: {aggregation}")


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    advantages: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: str = "grpo_clip",
    old_log_probs: Optional[torch.Tensor] = None,
    clip_eps: float = 0.2,
    aggregation: str = "seq_mean",
    normalize_constant: Optional[float] = None,
) -> tuple[torch.Tensor, dict]:
    """One RL micro-batch: per-token loss -> aggregate -> divide by grad-accum -> backward().

    Return (loss, {"clip_fraction": python float}) where ``loss`` is the scaled scalar
    you called backward on and clip_fraction is the fraction of RESPONSE tokens whose
    "clipped" flag is True (0.0 if there are no response tokens).
    """
    token_loss, metrics = per_token_policy_loss(
        policy_log_probs, advantages, old_log_probs, loss_type, clip_eps,
    )
    loss = aggregate_loss(token_loss, response_mask, aggregation, normalize_constant)
    loss = loss / gradient_accumulation_steps
    loss.backward()

    mask = response_mask.bool()
    clip_fraction = (metrics["clipped"] & mask).sum() / mask.sum().clamp_min(1)
    return loss, {"clip_fraction": clip_fraction.item()}
