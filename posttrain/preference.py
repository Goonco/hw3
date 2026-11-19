"""Part 3: preference optimization (DPO and IPO).

Implement every function marked TODO.  Tests: tests/test_preference.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sequence_log_probs(token_log_probs: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """log pi(y | x) for each sequence: the sum of token log-probs over response tokens.

    token_log_probs: (B, T); response_mask: (B, T)  ->  returns (B,)
    """
    return token_log_probs.masked_fill(~response_mask.bool(), 0.0).sum(dim=-1)


def preference_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
    loss_type: str = "dpo",
) -> tuple[torch.Tensor, dict]:
    """DPO / IPO loss on a batch of preference pairs.  All inputs have shape (B,).

    Let h = (policy_chosen - ref_chosen) - (policy_rejected - ref_rejected).

        loss_type == "dpo":  loss = mean_b[ -log sigmoid(beta * h) ]
        loss_type == "ipo":  loss = mean_b[ (h - 1 / (2 * beta))**2 ]
        anything else:       raise ValueError

    Do NOT call backward here.  Return (loss, metrics) where metrics is a dict of
    python floats (averaged over the batch):
        "chosen_reward":   mean of beta * (policy_chosen - ref_chosen)
        "rejected_reward": mean of beta * (policy_rejected - ref_rejected)
        "reward_margin":   mean of beta * h
        "reward_accuracy": fraction of pairs with h > 0 (strictly)
    """
    chosen_delta = policy_chosen_logps - ref_chosen_logps
    rejected_delta = policy_rejected_logps - ref_rejected_logps
    gap = chosen_delta - rejected_delta
    margin = beta * gap

    if loss_type == "dpo":
        loss = -F.logsigmoid(margin).mean()
    elif loss_type == "ipo":
        loss = (gap - 1 / (2 * beta)).square().mean()
    else:
        raise ValueError(f"Unknown preference loss: {loss_type}")

    return loss, {
        "chosen_reward": (beta * chosen_delta).mean().item(),
        "rejected_reward": (beta * rejected_delta).mean().item(),
        "reward_margin": margin.mean().item(),
        "reward_accuracy": (gap > 0).float().mean().item(),
    }
