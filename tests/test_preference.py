import pytest

from posttrain.preference import preference_loss, sequence_log_probs
from tests.helpers import assert_close, load_snapshot, t

METRIC_KEYS = ["chosen_reward", "rejected_reward", "reward_margin", "reward_accuracy"]


def test_sequence_log_probs():
    s = load_snapshot("preference")
    out = sequence_log_probs(t(s["token_log_probs"]), t(s["token_mask"]).bool())
    assert_close(out, s["expected_sequence_log_probs"], msg="sequence log-probs")


@pytest.mark.parametrize("loss_type", ["dpo", "ipo"])
def test_preference_loss(loss_type):
    s = load_snapshot("preference")
    pc = t(s["policy_chosen"], requires_grad=True)
    pr = t(s["policy_rejected"], requires_grad=True)
    loss, metrics = preference_loss(pc, pr, t(s["ref_chosen"]), t(s["ref_rejected"]), s["beta"], loss_type)
    exp = s[loss_type]
    assert_close(loss, exp["loss"], msg=f"{loss_type} loss")
    for k in METRIC_KEYS:
        assert k in metrics, f"missing metric {k}"
        assert_close(float(metrics[k]), exp[k], msg=f"{loss_type} metric {k}")
    loss.backward()
    assert_close(pc.grad, exp["grad_policy_chosen"], msg=f"{loss_type} grad w.r.t. policy_chosen_logps")
    assert_close(pr.grad, exp["grad_policy_rejected"], msg=f"{loss_type} grad w.r.t. policy_rejected_logps")


def test_preference_loss_rejects_unknown_type():
    s = load_snapshot("preference")
    args = [t(s[k]) for k in ("policy_chosen", "policy_rejected", "ref_chosen", "ref_rejected")]
    with pytest.raises(ValueError):
        preference_loss(*args, s["beta"], "not_a_loss")
