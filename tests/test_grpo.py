import pytest
import torch

from posttrain.grpo import aggregate_loss, compute_group_advantages, grpo_microbatch_train_step, per_token_policy_loss
from tests.helpers import assert_close, load_snapshot, t


@pytest.mark.parametrize("baseline", ["mean", "loo"])
@pytest.mark.parametrize("normalize_by_std", [True, False])
def test_compute_group_advantages(baseline, normalize_by_std):
    s = load_snapshot("advantages")
    key = f"{baseline}_{'std' if normalize_by_std else 'nostd'}"
    out = compute_group_advantages(s["rewards"], s["group_size"], baseline, normalize_by_std, s["eps"])
    assert_close(out, s["expected"][key], msg=key)


def test_compute_group_advantages_accepts_tensor():
    s = load_snapshot("advantages")
    out = compute_group_advantages(t(s["rewards"]), s["group_size"], "mean", True, s["eps"])
    assert_close(out, s["expected"]["mean_std"], msg="tensor input")


@pytest.mark.parametrize("loss_type", ["pg", "grpo_clip"])
def test_per_token_policy_loss(loss_type):
    s = load_snapshot("policy")
    loss, meta = per_token_policy_loss(
        t(s["log_probs"]), t(s["advantages"]), t(s["old_log_probs"]), loss_type, s["clip_eps"]
    )
    exp = s["per_token"][loss_type]
    assert_close(loss, exp["loss"], msg=f"{loss_type} per-token loss")
    assert meta["clipped"].bool().tolist() == exp["clipped"], "clipped mask mismatch"


@pytest.mark.parametrize("aggregation", ["seq_mean", "token_mean", "constant"])
def test_aggregate_loss(aggregation):
    s = load_snapshot("policy")
    out = aggregate_loss(t(s["aggregate_input"]), t(s["mask"]).bool(), aggregation, s["normalize_constant"])
    assert out.dim() == 0, "aggregate_loss must return a scalar"
    assert_close(out, s["aggregate"][aggregation], msg=aggregation)


@pytest.mark.parametrize("case", [0, 1, 2])
def test_grpo_microbatch_train_step(case):
    s = load_snapshot("policy")
    c = s["train_steps"][case]
    lp = t(s["log_probs"], requires_grad=True)
    loss, meta = grpo_microbatch_train_step(
        lp,
        t(s["mask"]).bool(),
        t(s["advantages"]),
        s["gradient_accumulation_steps"],
        loss_type=c["loss_type"],
        old_log_probs=t(s["old_log_probs"]),
        clip_eps=s["clip_eps"],
        aggregation=c["aggregation"],
        normalize_constant=s["normalize_constant"],
    )
    tag = f"{c['loss_type']}/{c['aggregation']}"
    assert_close(loss, c["expected_loss"], msg=f"{tag} loss")
    assert lp.grad is not None, "did you call loss.backward()?"
    assert_close(lp.grad, c["expected_grad"], msg=f"{tag} gradient")
    assert_close(float(meta["clip_fraction"]), c["expected_clip_fraction"], msg=f"{tag} clip_fraction")
