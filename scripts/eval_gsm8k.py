"""Greedy evaluation on the first 100 GSM8K test problems (Problem 7).  PROVIDED -- do not modify.

Writes <out>/predictions.jsonl and <out>/metrics.json.  The autograder re-scores
predictions.jsonl itself, so metrics.json is only a convenience copy.

    python scripts/eval_gsm8k.py --model Qwen/Qwen2.5-0.5B-Instruct --prompt standard --out results/gsm8k/A_standard
    python scripts/eval_gsm8k.py --model Qwen/Qwen2.5-0.5B-Instruct --prompt improved --out results/gsm8k/A_improved
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoTokenizer  # noqa: E402

from posttrain.generation import generate  # noqa: E402
from posttrain.gsm8k import (extract_answer, has_explicit_answer, improved_prompt_template,  # noqa: E402
                             is_correct, standard_prompt_template)
from posttrain.utils import DATA, env_info, read_jsonl, write_json, write_jsonl  # noqa: E402

MAX_NEW_TOKENS = 512  # fixed for grading
TEMPLATES = {"standard": standard_prompt_template, "improved": improved_prompt_template}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF model id or checkpoint dir")
    p.add_argument("--prompt", required=True, choices=sorted(TEMPLATES))
    p.add_argument("--out", required=True)
    p.add_argument("--split", default=str(DATA / "gsm8k_first_100.jsonl"))
    p.add_argument("--backend", default="auto", choices=["auto", "vllm", "hf"])
    p.add_argument("--limit", type=int, default=None, help="debug only; graded runs must use all 100 problems")
    args = p.parse_args()

    rows = read_jsonl(args.split)[: args.limit]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    template = TEMPLATES[args.prompt]
    prompts = [tokenizer.apply_chat_template(template(r["question"]), tokenize=False, add_generation_prompt=True)
               for r in rows]
    completions = generate(args.model, prompts, n=1, max_new_tokens=MAX_NEW_TOKENS, temperature=0.0,
                           backend=args.backend, batch_size=32)
    preds, n_correct, n_format = [], 0, 0
    for r, prompt, outs in zip(rows, prompts, completions):
        answer = extract_answer(outs[0])
        correct = is_correct(answer, r["answer"])
        n_correct += correct
        n_format += has_explicit_answer(outs[0])  # format rate: an explicit <answer> or \boxed answer
        preds.append({"id": r["id"], "prompt": prompt, "completion": outs[0], "answer": answer, "correct": correct})

    metrics = {
        "model": args.model,
        "prompt": args.prompt,
        "split": Path(args.split).name,
        "n": len(rows),
        "accuracy": n_correct / len(rows),
        "format_rate": n_format / len(rows),
        "max_new_tokens": MAX_NEW_TOKENS,
        "decoding": "greedy",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "env": env_info(),
    }
    write_jsonl(Path(args.out) / "predictions.jsonl", preds)
    write_json(Path(args.out) / "metrics.json", metrics)
    print(f"accuracy={metrics['accuracy']:.4f} format_rate={metrics['format_rate']:.4f} n={len(rows)}")


if __name__ == "__main__":
    main()
