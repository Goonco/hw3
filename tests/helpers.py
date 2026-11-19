"""Shared test helpers.  Snapshots are JSON files of inputs + expected outputs.

Set POSTTRAIN_SNAPSHOT_DIR to point the tests at a different snapshot folder
(the autograder uses hidden snapshots with different numbers).
"""

import json
import os
from pathlib import Path

import torch

SNAPSHOT_DIR = Path(os.environ.get("POSTTRAIN_SNAPSHOT_DIR", Path(__file__).resolve().parent / "snapshots"))


def load_snapshot(name: str) -> dict:
    with open(SNAPSHOT_DIR / f"{name}.json") as f:
        return json.load(f)


def t(x, dtype=torch.float32, requires_grad=False) -> torch.Tensor:
    return torch.tensor(x, dtype=dtype, requires_grad=requires_grad)


def assert_close(actual, expected, atol=1e-4, rtol=1e-4, msg=""):
    if not torch.is_tensor(actual):
        actual = torch.tensor(actual)
    expected = torch.tensor(expected, dtype=torch.float64)
    actual = actual.detach().to(torch.float64).cpu()
    assert actual.shape == expected.shape, f"{msg} shape {tuple(actual.shape)} != expected {tuple(expected.shape)}"
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, msg=lambda m: f"{msg}\n{m}")


class FakeTokenizer:
    """Character-level tokenizer with the subset of the HF API used in this assignment."""

    def __init__(self, vocab: dict, pad_token_id: int):
        self.vocab = vocab
        self.pad_token_id = pad_token_id
        self.eos_token_id = pad_token_id

    def encode(self, text, add_special_tokens=False):
        return [self.vocab[c] for c in text]

    def __call__(self, text, add_special_tokens=False, **kwargs):
        if isinstance(text, str):
            return {"input_ids": self.encode(text)}
        return {"input_ids": [self.encode(x) for x in text]}


class _Output:
    def __init__(self, logits):
        self.logits = logits


class FakeLM(torch.nn.Module):
    """logits[b, t, :] = table[input_ids[b, t]] + pos_bias[t]  (a tiny, fully known 'model')."""

    def __init__(self, table, pos_bias):
        super().__init__()
        self.table = torch.nn.Parameter(torch.tensor(table, dtype=torch.float32))
        self.register_buffer("pos_bias", torch.tensor(pos_bias, dtype=torch.float32))

    def forward(self, input_ids, **kwargs):
        return _Output(self.table[input_ids] + self.pos_bias[: input_ids.shape[1]].unsqueeze(0))
