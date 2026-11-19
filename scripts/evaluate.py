"""Greedy evaluation on the held-out Countdown test set.  PROVIDED -- do not modify.

Writes <out>/predictions.jsonl and <out>/metrics.json.  These two files are what the
autograder reads (it re-scores predictions.jsonl itself, so metrics.json is only a
convenience copy).  Decoding settings are fixed so everyone is graded identically.

    python scripts/evaluate.py --model results/sft/final --out results/sft
    python scripts/evaluate.py --model Qwen/Qwen2.5-0.5B --out results/base     # baseline
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from posttrain.countdown import score_response  # noqa: E402
from posttrain.generation import generate  # noqa: E402
from posttrain.utils import DATA, env_info, read_jsonl, write_json, write_jsonl  # noqa: E402

MAX_NEW_TOKENS = 384  # fixed for grading


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="checkpoint dir or HF model id")
    p.add_argument("--out", required=True)
    p.add_argument("--split", default=str(DATA / "countdown_test.jsonl"))
    p.add_argument("--backend", default="auto", choices=["auto", "vllm", "hf"])
    p.add_argument("--limit", type=int, default=None, help="debug only; graded runs must use the full split")
    args = p.parse_args()

    rows = read_jsonl(args.split)[: args.limit]
    completions = generate(args.model, [r["prompt"] for r in rows], n=1, max_new_tokens=MAX_NEW_TOKENS,
                           temperature=0.0, backend=args.backend, batch_size=64)
    preds, n_correct, n_format = [], 0, 0
    for r, outs in zip(rows, completions):
        s = score_response(outs[0], r["numbers"], r["target"])
        n_correct += s["correct"]
        n_format += s["format_ok"]
        preds.append({"id": r["id"], "completion": outs[0], "answer": s["answer"], "correct": s["correct"]})

    metrics = {
        "model": args.model,
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
