"""Package your submission.  PROVIDED.

    python scripts/make_submission.py --uniqname jdoe

Creates submission_<uniqname>.zip with your implementation and graded result files.  Missing
files are reported; the autograder gives 0 for anything missing.  The write-up PDF is submitted
separately to the "Homework 3 - Written" assignment on Gradescope.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "posttrain/common.py",
    "posttrain/sft.py",
    "posttrain/preference.py",
    "posttrain/grpo.py",
    "posttrain/test_time.py",
    "posttrain/gsm8k.py",
    "results/sft/metrics.json",
    "results/sft/predictions.jsonl",
    "results/dpo/metrics.json",
    "results/dpo/predictions.jsonl",
    "results/grpo/metrics.json",
    "results/grpo/predictions.jsonl",
    "results/test_time/test_time.json",
    "results/gsm8k/A_standard/predictions.jsonl",
    "results/gsm8k/B_standard/predictions.jsonl",
    "results/gsm8k/A_improved/predictions.jsonl",
]
OPTIONAL = [
    "results/test_time/test_time.png",
    "results/sft/train_log.jsonl",
    "results/dpo/train_log.jsonl",
    "results/grpo/train_log.jsonl",
    "results/sft/config.json",
    "results/dpo/config.json",
    "results/grpo/config.json",
    "results/base/metrics.json",
    "results/test_time_sft/test_time.json",
    "results/test_time_sft/test_time.png",
    "results/grpo_ablation/train_log.jsonl",
    "results/grpo_ablation/config.json",
    "results/gsm8k/A_standard/metrics.json",
    "results/gsm8k/B_standard/metrics.json",
    "results/gsm8k/A_improved/metrics.json",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--uniqname", required=True)
    args = p.parse_args()

    out = ROOT / f"submission_{args.uniqname}.zip"
    missing = []
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in REQUIRED + OPTIONAL:
            path = ROOT / rel
            if path.exists():
                z.write(path, rel)
            elif rel in REQUIRED:
                missing.append(rel)
    print(f"wrote {out}")
    if missing:
        print("WARNING, missing required files:\n  " + "\n  ".join(missing))


if __name__ == "__main__":
    main()
