"""Part 2: supervised fine-tuning on synthetic Countdown search traces.  PROVIDED.

Uses YOUR implementations of tokenize_prompt_and_output, get_response_log_probs and
sft_microbatch_train_step.

    python scripts/run_sft.py --out results/sft
    python scripts/evaluate.py --model results/sft/final --out results/sft
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from posttrain.common import get_response_log_probs, tokenize_prompt_and_output  # noqa: E402
from posttrain.sft import sft_microbatch_train_step  # noqa: E402
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


@torch.no_grad()
def validation_loss(model, tokenizer, rows, micro_batch_size) -> float:
    model.eval()
    total_nll, total_tokens = 0.0, 0
    for i in range(0, len(rows), micro_batch_size):
        mb = rows[i : i + micro_batch_size]
        enc = tokenize_prompt_and_output(
            [r["prompt"] for r in mb], [r["response"] + tokenizer.eos_token for r in mb], tokenizer
        )
        enc = {k: v.to(device()) for k, v in enc.items()}
        with autocast():
            lp = get_response_log_probs(model, enc["input_ids"], enc["labels"])["log_probs"].float()
        mask = enc["response_mask"].float()
        total_nll += -(lp * mask).sum().item()
        total_tokens += int(mask.sum().item())
    model.train()
    return total_nll / max(total_tokens, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--train", default=str(DATA / "sft_train.jsonl"))
    p.add_argument("--val", default=str(DATA / "sft_val.jsonl"))
    p.add_argument("--out", default="results/sft")
    p.add_argument("--max-examples", type=int, default=None, help="subsample the training set")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--micro-batch-size", type=int, default=8)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--grad-checkpointing", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    assert args.batch_size % args.micro_batch_size == 0
    grad_accum = args.batch_size // args.micro_batch_size

    set_seed(args.seed)
    out = Path(args.out)
    write_json(out / "config.json", {**vars(args), "env": env_info()})
    logger = JsonlLogger(out / "train_log.jsonl")

    model, tokenizer = load_model_and_tokenizer(args.model, gradient_checkpointing=args.grad_checkpointing)
    train_rows = read_jsonl(args.train)
    rng = random.Random(args.seed)
    if args.max_examples:
        train_rows = rng.sample(train_rows, args.max_examples)
    val_rows = read_jsonl(args.val)

    steps_per_epoch = len(train_rows) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    scheduler = warmup_cosine(optimizer, int(args.warmup_ratio * total_steps), total_steps)
    print(f"{len(train_rows)} examples, {total_steps} optimizer steps, grad_accum={grad_accum}")

    logger.log(step=0, val_loss=validation_loss(model, tokenizer, val_rows, args.micro_batch_size))
    model.train()
    step = 0
    for epoch in range(args.epochs):
        rng.shuffle(train_rows)
        for s in range(steps_per_epoch):
            batch = train_rows[s * args.batch_size : (s + 1) * args.batch_size]
            step_loss, n_tokens = 0.0, 0
            for m in range(grad_accum):
                mb = batch[m * args.micro_batch_size : (m + 1) * args.micro_batch_size]
                enc = tokenize_prompt_and_output(
                    [r["prompt"] for r in mb], [r["response"] + tokenizer.eos_token for r in mb], tokenizer
                )
                enc = {k: v.to(device()) for k, v in enc.items()}
                with autocast():
                    lp = get_response_log_probs(model, enc["input_ids"], enc["labels"])["log_probs"]
                mask = enc["response_mask"]
                # Per-example sum / (avg tokens per example)  ==  token-level mean over the micro-batch.
                normalize_constant = mask.sum().item() / mask.shape[0]
                loss, meta = sft_microbatch_train_step(lp.float(), mask, grad_accum, normalize_constant)
                step_loss += float(loss)
                n_tokens += meta["num_response_tokens"]
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            logger.log(step=step, epoch=epoch, train_loss=step_loss, grad_norm=grad_norm,
                       lr=scheduler.get_last_lr()[0], tokens=n_tokens)
            if step % args.eval_every == 0 or step == total_steps:
                logger.log(step=step, val_loss=validation_loss(model, tokenizer, val_rows, args.micro_batch_size))

    save_model(model, tokenizer, out / "final")


if __name__ == "__main__":
    main()
