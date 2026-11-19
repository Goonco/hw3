"""Generate the Countdown datasets used in this assignment.  PROVIDED.

The released data/ folder was produced with the default arguments; you do not need
to run this.  It is included so the data is fully reproducible.

Outputs (JSONL):
  data/countdown_test.jsonl   500 held-out problems (graded; never train on these)
  data/countdown_val.jsonl    200 problems for monitoring during training
  data/countdown_train.jsonl  RL prompts
  data/sft_train.jsonl        prompt + synthetic search-trace response
  data/sft_val.jsonl

SFT responses are synthetic "search traces": a few failed attempts followed by a
successful one, then the final <answer>.  They are generated programmatically, so
every SFT example is correct by construction.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from posttrain.countdown import (  # noqa: E402
    make_prompt,
    random_step_path,
    steps_to_expression,
    verify,
)


def sample_problem(rng: random.Random, n_numbers: int, lo=1, hi=99, tmin=10, tmax=100):
    while True:
        numbers = [rng.randint(lo, hi) for _ in range(n_numbers)]
        steps = random_step_path(numbers, rng)
        target = steps[-1][3]
        if tmin <= target <= tmax:
            return numbers, target, steps


def format_attempt(steps, target: int) -> str:
    parts = [f"{a} {op} {b} = {r}" for a, op, b, r in steps]
    result = steps[-1][3]
    body = "Try " + ", then ".join(parts) + "."
    if result == target:
        return body + " This equals the target!"
    return body + f" {result} != {target}, not correct."


def make_trace(numbers, target, solution_steps, rng: random.Random) -> str:
    n_fail = rng.choices([0, 1, 2, 3], weights=[3, 4, 3, 2])[0]
    lines = []
    tries = 0
    while len(lines) < n_fail and tries < 50:
        tries += 1
        steps = random_step_path(numbers, rng)
        if steps[-1][3] != target:
            lines.append(format_attempt(steps, target))
    lines.append(format_attempt(solution_steps, target))
    answer = steps_to_expression(numbers, solution_steps)
    assert verify(answer, numbers, target), (numbers, target, answer)
    return " " + "\n".join(lines) + "\n</think>\n<answer>" + answer + "</answer>"


def build(args):
    rng = random.Random(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seen: set = set()

    def fresh(n_numbers):
        while True:
            numbers, target, steps = sample_problem(rng, n_numbers)
            key = (tuple(sorted(numbers)), target)
            if key not in seen:
                seen.add(key)
                return numbers, target, steps

    def split(name: str, count: int, with_response: bool):
        rows = []
        for i in range(count):
            n_numbers = 3 if i % 2 == 0 else 4
            numbers, target, steps = fresh(n_numbers)
            row = {
                "id": f"{name}-{i:05d}",
                "numbers": numbers,
                "target": target,
                "prompt": make_prompt(numbers, target),
            }
            if with_response:
                row["response"] = make_trace(numbers, target, steps, rng)
            rows.append(row)
        return rows

    # Test set first, so it never changes when the other split sizes change.
    splits = {
        "countdown_test": split("test", args.n_test, False),
        "countdown_val": split("val", args.n_val, False),
        "sft_train": split("sft", args.n_sft, True),
        "sft_val": split("sftval", args.n_sft_val, True),
        "countdown_train": split("train", args.n_train, False),
    }
    for name, rows in splits.items():
        with open(out / f"{name}.jsonl", "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"wrote {len(rows):6d} rows to {out / (name + '.jsonl')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[1] / "data"))
    p.add_argument("--seed", type=int, default=224)
    p.add_argument("--n-test", type=int, default=500)
    p.add_argument("--n-val", type=int, default=200)
    p.add_argument("--n-sft", type=int, default=8000)
    p.add_argument("--n-sft-val", type=int, default=200)
    p.add_argument("--n-train", type=int, default=20000)
    build(p.parse_args())
