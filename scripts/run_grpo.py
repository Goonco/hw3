"""Part 4: RL with a verifiable reward (GRPO / RLOO / Dr. GRPO).  PROVIDED.

Rollouts are sampled with the in-memory policy (HF generate), scored by the Countdown
verifier, turned into advantages by YOUR compute_group_advantages, and optimised with
YOUR grpo_microbatch_train_step.

    python scripts/run_grpo.py --policy results/sft/final --out results/grpo
    python scripts/evaluate.py --model results/grpo/final --out results/grpo

Useful variants for the write-up:
    --baseline loo --no-std-normalization                     (RLOO)
    --aggregation constant --no-std-normalization             (Dr. GRPO)
    --epochs-per-rollout-batch 2                              (two optimizer steps per rollout batch)
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from posttrain.common import get_response_log_probs, masked_mean, tokenize_prompt_and_output  # noqa: E402
from posttrain.countdown import score_response  # noqa: E402
from posttrain.generation import hf_generate  # noqa: E402
from posttrain.grpo import compute_group_advantages, grpo_microbatch_train_step  # noqa: E402
from posttrain.utils import (  # noqa: E402
    DATA,
    JsonlLogger,
    autocast,
    device,
    env_info,
    load_model_and_tokenizer,
    read_jsonl,
    save_model,
    set_seed,
    write_json,
)


def greedy_accuracy(model, tokenizer, rows, max_new_tokens, batch_size=64) -> dict:
    outs = hf_generate(model, tokenizer, [r["prompt"] for r in rows], 1, max_new_tokens, 0.0, batch_size=batch_size)
    scores = [score_response(o[0], r["numbers"], r["target"]) for o, r in zip(outs, rows)]
    return {
        "val_accuracy": sum(s["correct"] for s in scores) / len(rows),
        "val_format_rate": sum(s["format_ok"] for s in scores) / len(rows),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="results/sft/final")
    p.add_argument("--prompts", default=str(DATA / "countdown_train.jsonl"))
    p.add_argument("--val", default=str(DATA / "countdown_val.jsonl"))
    p.add_argument("--out", default="results/grpo")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--prompts-per-step", type=int, default=32)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-6)
    p.add_argument("--micro-batch-size", type=int, default=16)
    p.add_argument("--epochs-per-rollout-batch", type=int, default=1)
    p.add_argument("--loss-type", choices=["pg", "grpo_clip"], default="grpo_clip")
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--baseline", choices=["mean", "loo"], default="mean")
    p.add_argument("--no-std-normalization", action="store_true")
    p.add_argument("--aggregation", choices=["seq_mean", "token_mean", "constant"], default="seq_mean")
    p.add_argument("--val-every", type=int, default=20)
    p.add_argument("--grad-checkpointing", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    n_rollouts = args.prompts_per_step * args.group_size
    assert n_rollouts % args.micro_batch_size == 0
    grad_accum = n_rollouts // args.micro_batch_size
    set_seed(args.seed)
    out = Path(args.out)
    write_json(out / "config.json", {**vars(args), "env": env_info()})
    logger = JsonlLogger(out / "train_log.jsonl")

    policy, tokenizer = load_model_and_tokenizer(args.policy, gradient_checkpointing=args.grad_checkpointing)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    train_rows = read_jsonl(args.prompts)
    val_rows = read_jsonl(args.val)
    rng = random.Random(args.seed)

    logger.log(step=0, **greedy_accuracy(policy, tokenizer, val_rows, args.max_new_tokens))
    for step in range(1, args.steps + 1):
        # 1) Rollouts: G samples for each of the step's prompts.
        rows = rng.sample(train_rows, args.prompts_per_step)
        samples = hf_generate(policy, tokenizer, [r["prompt"] for r in rows], args.group_size,
                              args.max_new_tokens, args.temperature)
        prompts, responses, rewards, correct, fmt = [], [], [], [], []
        for r, outs in zip(rows, samples):
            for completion in outs:
                s = score_response(completion, r["numbers"], r["target"])
                prompts.append(r["prompt"])
                responses.append(completion)
                rewards.append(s["reward"])
                correct.append(s["correct"])
                fmt.append(s["format_ok"])

        # 2) Advantages (your code).
        advantages = compute_group_advantages(
            rewards, args.group_size, baseline=args.baseline,
            normalize_by_std=not args.no_std_normalization,
        ).to(device())

        # 3) Tokenise once; cache old log-probs if we take several epochs / clip.
        micro = []
        for i in range(0, n_rollouts, args.micro_batch_size):
            enc = tokenize_prompt_and_output(prompts[i : i + args.micro_batch_size],
                                             responses[i : i + args.micro_batch_size], tokenizer)
            micro.append({k: v.to(device()) for k, v in enc.items()} | {"adv": advantages[i : i + args.micro_batch_size]})
        normalize_constant = float(args.max_new_tokens)
        if args.loss_type == "grpo_clip":
            policy.eval()
            with torch.no_grad():
                for mb in micro:
                    with autocast():
                        mb["old"] = get_response_log_probs(policy, mb["input_ids"], mb["labels"])["log_probs"].float()
            policy.train()

        # 4) Policy update(s).  (Entropy is measured on the first micro-batch, without grad.)
        with torch.no_grad(), autocast():
            ent = get_response_log_probs(policy, micro[0]["input_ids"], micro[0]["labels"], return_token_entropy=True)
        entropy = masked_mean(ent["token_entropy"].float(), micro[0]["response_mask"]).item()
        del ent
        policy.train()
        clip_fracs = []
        for _ in range(args.epochs_per_rollout_batch):
            for mb in micro:
                with autocast():
                    log_probs = get_response_log_probs(policy, mb["input_ids"], mb["labels"])["log_probs"]
                _, meta = grpo_microbatch_train_step(
                    log_probs.float(), mb["response_mask"], mb["adv"], grad_accum,
                    loss_type=args.loss_type, old_log_probs=mb.get("old"), clip_eps=args.clip_eps,
                    aggregation=args.aggregation, normalize_constant=normalize_constant,
                )
                clip_fracs.append(meta["clip_fraction"])
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0).item()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        logger.log(
            step=step,
            reward=statistics.mean(rewards),
            train_accuracy=statistics.mean(correct),
            format_rate=statistics.mean(fmt),
            response_chars=statistics.mean([len(x) for x in responses]),
            frac_groups_all_same=sum(len(set(rewards[i : i + args.group_size])) == 1
                                     for i in range(0, n_rollouts, args.group_size)) / args.prompts_per_step,
            entropy=entropy,
            clip_fraction=statistics.mean(clip_fracs),
            grad_norm=grad_norm,
        )
        if step % args.val_every == 0 or step == args.steps:
            logger.log(step=step, **greedy_accuracy(policy, tokenizer, val_rows, args.max_new_tokens))

    save_model(policy, tokenizer, out / "final")


if __name__ == "__main__":
    main()
