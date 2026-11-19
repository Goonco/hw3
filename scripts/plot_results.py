"""Make every plot and number the write-up asks for.  PROVIDED -- run it on a login node (CPU only).

    python scripts/plot_results.py            # reads results/, writes the PNGs listed below

    results/sft/loss.png                   Problem 2  SFT training and validation loss
    results/dpo/rewards.png                Problem 3  validation reward accuracy and reward margin
    results/grpo/curves.png                Problem 4  reward, validation accuracy, entropy, response length,
                                                      frac_groups_all_same, clip_fraction
    results/grpo_ablation/compare.png      Problem 4  your ablation vs the first 40 steps of the default run
    results/gsm8k/bar.png                  Problem 7  GSM8K accuracy: A standard, B standard, A improved

It also prints a table of accuracy / format rate for every run it finds.  A plot whose inputs
do not exist yet is skipped with a message, so you can run this after each stage.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def read_log(path: Path) -> list[dict]:
    """train_log.jsonl rows of the LAST run (the logger appends, so a re-run restarts at step 0)."""
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    start = 0
    for i in range(1, len(rows)):
        if rows[i].get("step", 0) < rows[i - 1].get("step", 0):
            start = i
    return rows[start:]


def series(rows: list[dict], key: str, max_step: float = float("inf")) -> tuple[list, list]:
    pts = [(r["step"], r[key]) for r in rows if key in r and r["step"] <= max_step]
    return [s for s, _ in pts], [v for _, v in pts]


def save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path.relative_to(ROOT)}")


def need(*paths: Path) -> bool:
    missing = [str(p.relative_to(ROOT)) for p in paths if not p.exists()]
    if missing:
        print(f"skipped (missing {', '.join(missing)})")
    return not missing


def plot_sft() -> None:
    log = RESULTS / "sft" / "train_log.jsonl"
    if not need(log):
        return
    rows = read_log(log)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(*series(rows, "train_loss"), lw=0.8, alpha=0.7, label="training loss (per step)")
    ax.plot(*series(rows, "val_loss"), marker="o", label="validation loss")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("loss (per response token)")
    ax.set_title("SFT")
    ax.grid(alpha=0.3)
    ax.legend()
    save(fig, RESULTS / "sft" / "loss.png")


def plot_dpo() -> None:
    log = RESULTS / "dpo" / "train_log.jsonl"
    if not need(log):
        return
    rows = read_log(log)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, key, label in [(axes[0], "reward_accuracy", "reward accuracy"),
                           (axes[1], "reward_margin", "reward margin ($\\beta h$)")]:
        ax.plot(*series(rows, key), lw=0.8, alpha=0.5, label="training batch")
        ax.plot(*series(rows, "val_" + key), marker="o", label="validation")
        ax.set_xlabel("optimizer step")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("DPO")
    save(fig, RESULTS / "dpo" / "rewards.png")


GRPO_PANELS = [
    ("reward", "mean reward of the samples"),
    ("val_accuracy", "validation accuracy (greedy)"),
    ("entropy", "mean token entropy (nats)"),
    ("response_chars", "mean response length (chars)"),
    ("frac_groups_all_same", "frac. groups with equal rewards"),
    ("clip_fraction", "clip fraction"),
]


def grpo_figure(runs: list[tuple[str, list[dict]]], max_step: float, title: str):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, (key, label) in zip(axes.flat, GRPO_PANELS):
        for name, rows in runs:
            x, y = series(rows, key, max_step)
            ax.plot(x, y, marker="o" if key.startswith("val_") else None, label=name)
        ax.set_xlabel("GRPO step")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    if len(runs) > 1:
        axes.flat[0].legend()
    fig.suptitle(title)
    return fig


def plot_grpo() -> None:
    log = RESULTS / "grpo" / "train_log.jsonl"
    if not need(log):
        return
    save(grpo_figure([("default", read_log(log))], float("inf"), "GRPO (default run)"), RESULTS / "grpo" / "curves.png")


def plot_ablation() -> None:
    log, abl = RESULTS / "grpo" / "train_log.jsonl", RESULTS / "grpo_ablation" / "train_log.jsonl"
    if not need(log, abl):
        return
    name = "ablation"
    cfg = RESULTS / "grpo_ablation" / "config.json"
    if cfg.exists():
        c = json.loads(cfg.read_text())
        changed = [f"baseline={c.get('baseline')}" if c.get("baseline", "mean") != "mean" else "",
                   f"aggregation={c.get('aggregation')}" if c.get("aggregation", "seq_mean") != "seq_mean" else "",
                   "no std norm." if c.get("no_std_normalization") else "",
                   f"{c.get('epochs_per_rollout_batch')} epochs/batch" if c.get("epochs_per_rollout_batch", 1) != 1 else ""]
        name = "ablation: " + (", ".join(x for x in changed if x) or "default settings")
    abl_rows = read_log(abl)
    last = max(r["step"] for r in abl_rows)
    fig = grpo_figure([("default", read_log(log)), (name, abl_rows)], last, f"GRPO: default vs ablation, steps 0-{last}")
    save(fig, RESULTS / "grpo_ablation" / "compare.png")


GSM8K_RUNS = [("A_standard", "model A\nstandard prompt"), ("B_standard", "model B\nstandard prompt"),
              ("A_improved", "model A\nyour prompt")]


def plot_gsm8k() -> None:
    paths = [RESULTS / "gsm8k" / run / "metrics.json" for run, _ in GSM8K_RUNS]
    if not need(*paths):
        return
    accs = [json.loads(p.read_text())["accuracy"] for p in paths]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar([label for _, label in GSM8K_RUNS], accs, color=["tab:blue", "tab:orange", "tab:green"])
    ax.bar_label(bars, labels=[f"{a:.2f}" for a in accs])
    ax.set_ylim(0, 1)
    ax.set_ylabel("accuracy on 100 GSM8K problems")
    ax.set_title("GSM8K (greedy)")
    ax.grid(axis="y", alpha=0.3)
    save(fig, RESULTS / "gsm8k" / "bar.png")


def summary_table() -> None:
    runs = ["base", "sft", "dpo", "grpo"] + [f"gsm8k/{r}" for r, _ in GSM8K_RUNS]
    print(f"\n{'run':<20}{'accuracy':>10}{'format rate':>14}")
    for run in runs:
        path = RESULTS / run / "metrics.json"
        if path.exists():
            m = json.loads(path.read_text())
            print(f"{run:<20}{m['accuracy']:>10.3f}{m['format_rate']:>14.3f}")
        else:
            print(f"{run:<20}{'(missing)':>10}")


def main() -> None:
    if not RESULTS.is_dir():
        sys.exit(f"no results/ folder at {RESULTS}")
    for fn in (plot_sft, plot_dpo, plot_grpo, plot_ablation, plot_gsm8k):
        print(f"[{fn.__name__}] ", end="")
        fn()
    summary_table()


if __name__ == "__main__":
    main()
