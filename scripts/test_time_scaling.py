"""Part 5: test-time scaling.  PROVIDED.

Samples N responses per test problem ONCE, then estimates, for each budget k <= N:

  pass@k  - at least one of k samples is correct (needs an oracle verifier; an upper bound)
  maj@k   - majority vote over the VALUE of each sample's answer expression, then check
            the first sample that reached the voted value (no verifier needed)
  conf@k  - pick the sample with the highest mean token log-prob (no verifier needed)

maj@k and conf@k are averaged over random size-k subsets of the N samples.
Uses YOUR pass_at_k, majority_vote and select_by_confidence.

    python scripts/test_time_scaling.py --model results/grpo/final --out results/test_time
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from posttrain.countdown import answer_value, score_response  # noqa: E402
from posttrain.generation import generate  # noqa: E402
from posttrain.test_time import majority_vote, pass_at_k, select_by_confidence  # noqa: E402
from posttrain.utils import DATA, env_info, read_jsonl, write_json, write_jsonl  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", default="results/test_time")
    p.add_argument("--split", default=str(DATA / "countdown_test.jsonl"))
    p.add_argument("--num-problems", type=int, default=200)
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--num-subsets", type=int, default=64)
    p.add_argument("--backend", default="auto")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    assert max(args.ks) <= args.n

    rows = read_jsonl(args.split)[: args.num_problems]
    texts, conf = generate(args.model, [r["prompt"] for r in rows], args.n, args.max_new_tokens,
                           args.temperature, seed=args.seed, backend=args.backend, return_confidence=True)

    per_problem = []
    for r, outs, scores in zip(rows, texts, conf):
        judged = [score_response(o, r["numbers"], r["target"]) for o in outs]
        per_problem.append({
            "id": r["id"],
            "correct": [j["correct"] for j in judged],
            "values": [answer_value(j["answer"]) for j in judged],
            "confidence": scores,
            "completions": outs,
        })

    rng = random.Random(args.seed)
    curves = {"k": args.ks, "pass_at_k": [], "maj_at_k": [], "conf_at_k": []}
    for k in args.ks:
        pk, mk, ck = [], [], []
        for prob in per_problem:
            n, c = len(prob["correct"]), sum(prob["correct"])
            pk.append(pass_at_k(n, c, k))
            subsets = [list(range(n))] if k == n else [rng.sample(range(n), k) for _ in range(args.num_subsets)]
            m_hits = c_hits = 0
            for sub in subsets:
                voted = majority_vote([prob["values"][i] for i in sub])
                if voted is not None:
                    pick = next(i for i in sub if prob["values"][i] == voted)
                    m_hits += prob["correct"][pick]
                pick = sub[select_by_confidence([prob["confidence"][i] for i in sub])]
                c_hits += prob["correct"][pick]
            mk.append(m_hits / len(subsets))
            ck.append(c_hits / len(subsets))
        curves["pass_at_k"].append(sum(pk) / len(pk))
        curves["maj_at_k"].append(sum(mk) / len(mk))
        curves["conf_at_k"].append(sum(ck) / len(ck))
        print(f"k={k:3d} pass@k={curves['pass_at_k'][-1]:.3f} maj@k={curves['maj_at_k'][-1]:.3f} "
              f"conf@k={curves['conf_at_k'][-1]:.3f}")

    out = Path(args.out)
    write_json(out / "test_time.json", {**curves, "model": args.model, "n": args.n,
                                         "num_problems": len(rows), "temperature": args.temperature,
                                         "env": env_info()})
    write_jsonl(out / "samples.jsonl", per_problem)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 3.5))
        for key, label in [("pass_at_k", "pass@k (oracle)"), ("maj_at_k", "majority vote"), ("conf_at_k", "max confidence")]:
            ax.plot(curves["k"], curves[key], marker="o", label=label)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("samples per problem (k)")
        ax.set_ylabel("accuracy")
        ax.set_title(Path(args.model).parent.name or args.model)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "test_time.png", dpi=150)
        print(f"saved plot to {out / 'test_time.png'}")
    except ImportError:
        print("matplotlib not installed; skipping plot")


if __name__ == "__main__":
    main()
