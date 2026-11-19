"""Countdown task utilities.  PROVIDED -- do not modify.

A Countdown problem gives a list of integers and a target.  The model must write an
arithmetic expression that uses every given number exactly once, with only + - * /
and parentheses, and evaluates to the target.

Everything here is pure Python (no torch), so the autograder can re-score your
predictions on a CPU.
"""

from __future__ import annotations

import ast
import random
import re
from fractions import Fraction
from typing import Optional

STOP_STRING = "</answer>"

PROMPT_TEMPLATE = (
    "A conversation between User and Assistant. The user asks a question, and the "
    "assistant solves it. The assistant first thinks about the reasoning process step "
    "by step and then provides the user with the answer.\n"
    "User: Using the numbers {numbers}, create an equation that equals {target}. "
    "You can use basic arithmetic operations (+, -, *, /) and each number must be used "
    "exactly once. Show your work in <think> </think> tags and return the final equation "
    "in <answer> </answer> tags, for example <answer>(1 + 2) / 3</answer>.\n"
    "Assistant: Let me solve this step by step.\n<think>"
)


def make_prompt(numbers: list[int], target: int) -> str:
    return PROMPT_TEMPLATE.format(numbers=list(numbers), target=target)


# --------------------------------------------------------------------------------------
# Parsing and exact evaluation
# --------------------------------------------------------------------------------------

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_ALLOWED_CHARS = re.compile(r"^[\d+\-*/().\s]+$")
_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
}


def extract_answer(response: str) -> Optional[str]:
    """Return the contents of the LAST <answer>...</answer> block, or None."""
    matches = _ANSWER_RE.findall(response)
    if not matches:
        return None
    return matches[-1].strip()


def _eval_node(node: ast.AST) -> Fraction:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return Fraction(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Div) and right == 0:
            raise ZeroDivisionError
        return _BINOPS[type(node.op)](left, right)
    # Unary minus, floats, names, calls, ... are all rejected.
    raise ValueError(f"disallowed syntax: {ast.dump(node)[:80]}")


def evaluate_equation(expr: Optional[str]) -> Optional[Fraction]:
    """Exactly evaluate an expression with rational arithmetic.  None if invalid.

    A trailing "= <number>" is tolerated (e.g. "3 * 4 = 12" evaluates the left side).
    """
    if expr is None:
        return None
    expr = expr.split("=")[0].strip()
    if not expr or len(expr) > 200 or not _ALLOWED_CHARS.match(expr):
        return None
    try:
        return _eval_node(ast.parse(expr, mode="eval"))
    except (SyntaxError, ValueError, ZeroDivisionError, RecursionError):
        return None


def numbers_in(expr: str) -> list[int]:
    return [int(x) for x in re.findall(r"\d+", expr.split("=")[0])]


def verify(expr: Optional[str], numbers: list[int], target: int) -> bool:
    """True iff expr uses each given number exactly once and evaluates to target."""
    if expr is None:
        return False
    value = evaluate_equation(expr)
    if value is None:
        return False
    return sorted(numbers_in(expr)) == sorted(numbers) and value == target


def answer_value(expr: Optional[str]) -> Optional[str]:
    """Canonical string of the expression's value (e.g. "98" or "7/3"); None if invalid.

    Useful as a hashable key for majority voting.
    """
    value = evaluate_equation(expr)
    return None if value is None else str(value)


def score_response(response: str, numbers: list[int], target: int) -> dict:
    """Rule-based reward used for RL and evaluation.

    reward = 1.0 if the final answer is correct,
             0.1 if an <answer> block exists but is wrong (format reward),
             0.0 otherwise.
    """
    expr = extract_answer(response)
    format_ok = expr is not None
    correct = verify(expr, numbers, target)
    reward = 1.0 if correct else (0.1 if format_ok else 0.0)
    return {
        "reward": reward,
        "format_ok": format_ok,
        "correct": correct,
        "answer": expr,
    }


# --------------------------------------------------------------------------------------
# Brute-force solver (used for data generation and sanity checks)
# --------------------------------------------------------------------------------------


def _strip_outer_parens(expr: str) -> str:
    if not (expr.startswith("(") and expr.endswith(")")):
        return expr
    depth = 0
    for i, ch in enumerate(expr):
        depth += ch == "("
        depth -= ch == ")"
        if depth == 0 and i < len(expr) - 1:
            return expr  # the first "(" closes before the end
    return expr[1:-1]


def solve(numbers: list[int], target: int) -> Optional[str]:
    """Return some expression reaching target using all numbers, or None."""
    target_f = Fraction(target)

    def rec(items: list[tuple[Fraction, str]]) -> Optional[str]:
        if len(items) == 1:
            return items[0][1] if items[0][0] == target_f else None
        for i in range(len(items)):
            for j in range(len(items)):
                if i == j:
                    continue
                (a, ea), (b, eb) = items[i], items[j]
                rest = [items[k] for k in range(len(items)) if k not in (i, j)]
                candidates = [(a - b, f"({ea} - {eb})")]
                if i < j:
                    candidates += [(a + b, f"({ea} + {eb})"), (a * b, f"({ea} * {eb})")]
                if b != 0:
                    candidates.append((a / b, f"({ea} / {eb})"))
                for value, expr in candidates:
                    found = rec(rest + [(value, expr)])
                    if found is not None:
                        return found
        return None

    found = rec([(Fraction(n), str(n)) for n in numbers])
    return None if found is None else _strip_outer_parens(found)


def random_step_path(numbers: list[int], rng: random.Random) -> Optional[list[tuple[int, str, int, int]]]:
    """Randomly combine numbers two at a time, keeping every intermediate a positive
    integer (like the TV game).  Returns [(a, op, b, result), ...] or None."""
    pool = list(numbers)
    steps = []
    while len(pool) > 1:
        i, j = rng.sample(range(len(pool)), 2)
        a, b = pool[i], pool[j]
        options = [("+", a + b), ("*", a * b)]
        if a - b > 0:
            options.append(("-", a - b))
        if b > 1 and a % b == 0:
            options.append(("/", a // b))
        op, result = rng.choice(options)
        steps.append((a, op, b, result))
        pool = [pool[k] for k in range(len(pool)) if k not in (i, j)] + [result]
    return steps


def steps_to_expression(numbers: list[int], steps: list[tuple[int, str, int, int]]) -> str:
    """Turn a step path into a single parenthesised expression."""
    pool: list[tuple[int, str]] = [(n, str(n)) for n in numbers]
    for a, op, b, result in steps:
        ia = next(k for k, (v, _) in enumerate(pool) if v == a)
        ea = pool.pop(ia)[1]
        ib = next(k for k, (v, _) in enumerate(pool) if v == b)
        eb = pool.pop(ib)[1]
        pool.append((result, f"({ea} {op} {eb})"))
    return _strip_outer_parens(pool[0][1])
