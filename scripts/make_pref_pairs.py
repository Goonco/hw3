"""Build preference pairs for DPO/IPO.  PROVIDED.

--mode onpolicy (default): sample K responses per prompt from your SFT model, label
    them with the Countdown verifier, and pair a correct response (chosen) with an
    incorrect one (rejected).  This is "RL from AI/verifier feedback" in its simplest
    form: the verifier plays the role of the human labeller.
    With --greedy-predictions (the output of scripts/evaluate.py on training prompts),
    only prompts that the model gets WRONG under greedy decoding are used, and the
    rejected response is that greedy completion -- the same kind of mistake the graded
    (greedy) evaluation sees.  The chosen response is still a correct sample.
--mode synthetic: no GPU needed.  chosen = a gold search trace, rejected = the same
    trace with a corrupted final answer.  Use only if you are short on compute
    (and discuss the difference in your write-up).

    python scripts/make_pref_pairs.py --model results/sft/final --out data/pref_pairs.jsonl

    python scripts/evaluate.py --model results/sft/final --split data/countdown_train.jsonl --limit 7000 \
        --out results/sft_train_greedy
    python scripts/make_pref_pairs.py --model results/sft/final \
        --greedy-predictions results/sft_train_greedy/predictions.jsonl --out data/pref_pairs.jsonl
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from posttrain.countdown import extract_answer, score_response  # noqa: E402
from posttrain.utils import DATA, read_jsonl, write_jsonl  # noqa: E402


def corrupt(response: str, numbers: list[int], rng: random.Random) -> str:
    answer = extract_answer(response)
    options = [
        answer.replace("+", "-", 1) if "+" in answer else answer.replace("*", "+", 1),
        answer.replace(str(numbers[0]), str(numbers[0] + rng.randint(1, 9)), 1),
        " + ".join(str(n) for n in numbers),
    ]
    bad = rng.choice(options)
    return response.replace(f"<answer>{answer}</answer>", f"<answer>{bad}</answer>")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["onpolicy", "synthetic"], default="onpolicy")
    p.add_argument("--model", default="results/sft/final")
    p.add_argument("--prompts", default=str(DATA / "countdown_train.jsonl"))
    p.add_argument("--sft-data", default=str(DATA / "sft_train.jsonl"))
    p.add_argument("--num-prompts", type=int, default=3000)
    p.add_argument("--samples-per-prompt", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--backend", default="auto")
    p.add_argument("--greedy-predictions", default=None,
                   help="predictions.jsonl from evaluate.py on training prompts; rejected = wrong greedy completion")
    p.add_argument("--out", default=str(DATA / "pref_pairs.jsonl"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = random.Random(args.seed)

    pairs = []
    if args.mode == "synthetic":
        rows = rng.sample(read_jsonl(args.sft_data), args.num_prompts)
        for r in rows:
            rejected = corrupt(r["response"], r["numbers"], rng)
            if not score_response(rejected, r["numbers"], r["target"])["correct"]:
                pairs.append({"id": r["id"], "prompt": r["prompt"], "chosen": r["response"], "rejected": rejected})
    else:
        from posttrain.generation import generate

        greedy = None
        if args.greedy_predictions:
            greedy = {g["id"]: g["completion"] for g in read_jsonl(args.greedy_predictions)}
            by_id = {r["id"]: r for r in read_jsonl(args.prompts)}
            rows = [by_id[i] for i, c in greedy.items()
                    if not score_response(c, by_id[i]["numbers"], by_id[i]["target"])["correct"]]
            print(f"greedy completions: {len(greedy)}, wrong: {len(rows)}")
        else:
            rows = rng.sample(read_jsonl(args.prompts), args.num_prompts)
        completions = generate(
            args.model, [r["prompt"] for r in rows], args.samples_per_prompt, args.max_new_tokens,
            args.temperature, seed=args.seed, backend=args.backend,
        )
        n_all_right = n_all_wrong = 0
        for r, outs in zip(rows, completions):
            scored = [(c, score_response(c, r["numbers"], r["target"])) for c in outs]
            good = [c for c, s in scored if s["correct"]]
            bad = [c for c, s in scored if not s["correct"]]
            bad_formatted = [c for c, s in scored if not s["correct"] and s["format_ok"]]
            if not good:
                n_all_wrong += 1
                continue
            if greedy is not None:
                pairs.append({"id": r["id"], "prompt": r["prompt"], "chosen": rng.choice(good), "rejected": greedy[r["id"]]})
                continue
            if not bad:
                n_all_right += 1
                continue
            pairs.append({
                "id": r["id"],
                "prompt": r["prompt"],
                "chosen": rng.choice(good),
                "rejected": rng.choice(bad_formatted or bad),
            })
        print(f"prompts with all samples correct: {n_all_right}, all wrong: {n_all_wrong}")

    write_jsonl(args.out, pairs)
    print(f"wrote {len(pairs)} preference pairs to {args.out}")


if __name__ == "__main__":
    main()
