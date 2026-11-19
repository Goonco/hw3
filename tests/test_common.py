import torch

from posttrain.common import (
    compute_entropy,
    get_response_log_probs,
    masked_mean,
    masked_normalize,
    tokenize_prompt_and_output,
)
from tests.helpers import FakeLM, FakeTokenizer, assert_close, load_snapshot, t


def test_tokenize_prompt_and_output():
    s = load_snapshot("tokenize")
    tok = FakeTokenizer(s["vocab"], s["pad_token_id"])
    out = tokenize_prompt_and_output(s["prompts"], s["outputs"], tok)
    exp = s["expected"]
    assert set(out) >= {"input_ids", "labels", "response_mask"}
    assert out["input_ids"].dtype == torch.long and out["labels"].dtype == torch.long
    assert out["input_ids"].tolist() == exp["input_ids"], "input_ids mismatch"
    assert out["labels"].tolist() == exp["labels"], "labels mismatch"
    assert out["response_mask"].long().tolist() == exp["response_mask"], "response_mask mismatch"


def test_compute_entropy():
    s = load_snapshot("entropy")
    assert_close(compute_entropy(t(s["logits"])), s["expected"], msg="entropy")


def test_compute_entropy_is_numerically_stable():
    s = load_snapshot("entropy")
    out = compute_entropy(t(s["logits_large"]))
    assert torch.isfinite(out).all(), "entropy produced nan/inf for large logits"
    assert_close(out, s["expected_large"], msg="entropy (large logits)")


def test_get_response_log_probs():
    s = load_snapshot("logprobs")
    model = FakeLM(s["table"], s["pos_bias"])
    ids, labels = t(s["input_ids"], torch.long), t(s["labels"], torch.long)
    out = get_response_log_probs(model, ids, labels, return_token_entropy=False)
    assert_close(out["log_probs"], s["expected_log_probs"], msg="log_probs")


def test_get_response_log_probs_with_entropy_and_grad():
    s = load_snapshot("logprobs")
    model = FakeLM(s["table"], s["pos_bias"])
    ids, labels = t(s["input_ids"], torch.long), t(s["labels"], torch.long)
    out = get_response_log_probs(model, ids, labels, return_token_entropy=True)
    assert_close(out["log_probs"], s["expected_log_probs"], msg="log_probs")
    assert_close(out["token_entropy"], s["expected_entropy"], msg="token_entropy")
    out["log_probs"].sum().backward()  # log-probs must stay connected to the model parameters
    assert model.table.grad is not None and model.table.grad.abs().sum() > 0


def test_masked_mean():
    s = load_snapshot("masked")
    x, m = t(s["tensor"]), t(s["mask"], torch.bool)
    assert_close(masked_mean(x, m), s["mean_all"], msg="dim=None")
    assert_close(masked_mean(x, m, dim=0), s["mean_dim0"], msg="dim=0")
    assert_close(masked_mean(x, m, dim=-1), s["mean_dim_last"], msg="dim=-1")
    assert_close(masked_mean(x, m.float(), dim=-1), s["mean_dim_last"], msg="float mask")


def test_masked_normalize():
    s = load_snapshot("masked")
    x, m, c = t(s["tensor"]), t(s["mask"], torch.bool), s["normalize_constant"]
    assert_close(masked_normalize(x, m, c), s["normalize_all"], msg="dim=None")
    assert_close(masked_normalize(x, m, c, dim=0), s["normalize_dim0"], msg="dim=0")
    assert_close(masked_normalize(x, m, c, dim=-1), s["normalize_dim_last"], msg="dim=-1")
