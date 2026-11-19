"""Part 3: preference optimisation (DPO or IPO) starting from your SFT model.  PROVIDED.

Uses YOUR tokenize_prompt_and_output, get_response_log_probs, sequence_log_probs and
preference_loss.

    python scripts/run_dpo.py --policy results/sft/final --pairs data/pref_pairs.jsonl --out results/dpo
    python scripts/evaluate.py --model results/dpo/final --out results/dpo
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from posttrain.common import get_response_log_probs, tokenize_prompt_and_output  # noqa: E402
from posttrain.preference import preference_loss, sequence_log_probs  # noqa: E402
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
    warmup_cosine,
    write_json,
)


def pair_logps(model, tokenizer, rows):
    """Sequence log-probs of chosen and rejected responses (one forward pass on 2B sequences)."""
    prompts = [r["prompt"] for r in rows] * 2
    outputs = [r["chosen"] for r in rows] + [r["rejected"] for r in rows]
    enc = tokenize_prompt_and_output(prompts, outputs, tokenizer)
    enc = {k: v.to(device()) for k, v in enc.items()}
    with autocast():
        lp = get_response_log_probs(model, enc["input_ids"], enc["labels"])["log_probs"].float()
    seq = sequence_log_probs(lp, enc["response_mask"])
    return seq[: len(rows)], seq[len(rows) :]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="results/sft/final")
    p.add_argument("--pairs", default=str(DATA / "pref_pairs.jsonl"))
    p.add_argument("--out", default="results/dpo")
    p.add_argument("--loss-type", choices=["dpo", "ipo"], default="dpo")
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-7)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--micro-batch-size", type=int, default=4)
    p.add_argument("--val-pairs", type=int, default=128, help="held out from --pairs for monitoring")
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--grad-checkpointing", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    grad_accum = args.batch_size // args.micro_batch_size

    set_seed(args.seed)
    out = Path(args.out)
    write_json(out / "config.json", {**vars(args), "env": env_info()})
    logger = JsonlLogger(out / "train_log.jsonl")

    policy, tokenizer = load_model_and_tokenizer(args.policy, gradient_checkpointing=args.grad_checkpointing)
    reference, _ = load_model_and_tokenizer(args.policy, dtype=torch.bfloat16)
    reference.eval().requires_grad_(False)

    rows = read_jsonl(args.pairs)
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    val_rows, train_rows = rows[: args.val_pairs], rows[args.val_pairs :]
    total_steps = (len(train_rows) // args.batch_size) * args.epochs
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    scheduler = warmup_cosine(optimizer, max(1, total_steps // 20), total_steps)
    print(f"{len(train_rows)} train pairs, {len(val_rows)} val pairs, {total_steps} steps")

    @torch.no_grad()
    def validate(step):
        policy.eval()
        agg = {}
        for i in range(0, len(val_rows), args.micro_batch_size):
            mb = val_rows[i : i + args.micro_batch_size]
            pc, pr = pair_logps(policy, tokenizer, mb)
            rc, rr = pair_logps(reference, tokenizer, mb)
            loss, metrics = preference_loss(pc, pr, rc, rr, args.beta, args.loss_type)
            for k, v in {"loss": float(loss), **metrics}.items():
                agg[k] = agg.get(k, 0.0) + float(v) * len(mb)
        policy.train()
        logger.log(step=step, **{f"val_{k}": v / len(val_rows) for k, v in agg.items()})

    validate(0)
    policy.train()
    step = 0
    for epoch in range(args.epochs):
        rng.shuffle(train_rows)
        for s in range(len(train_rows) // args.batch_size):
            batch = train_rows[s * args.batch_size : (s + 1) * args.batch_size]
            agg = {}
            for m in range(grad_accum):
                mb = batch[m * args.micro_batch_size : (m + 1) * args.micro_batch_size]
                with torch.no_grad():
                    rc, rr = pair_logps(reference, tokenizer, mb)
                pc, pr = pair_logps(policy, tokenizer, mb)
                loss, metrics = preference_loss(pc, pr, rc, rr, args.beta, args.loss_type)
                (loss / grad_accum).backward()
                for k, v in {"loss": float(loss), **metrics}.items():
                    agg[k] = agg.get(k, 0.0) + float(v) / grad_accum
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0).item()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            logger.log(step=step, epoch=epoch, grad_norm=grad_norm, lr=scheduler.get_last_lr()[0], **agg)
            if step % args.eval_every == 0 or step == total_steps:
                validate(step)

    save_model(policy, tokenizer, out / "final")


if __name__ == "__main__":
    main()
