from posttrain.sft import sft_microbatch_train_step
from tests.helpers import assert_close, load_snapshot, t


def test_sft_microbatch_train_step_loss_and_grad():
    s = load_snapshot("sft")
    lp = t(s["log_probs"], requires_grad=True)
    mask = t(s["mask"]).bool()
    loss, meta = sft_microbatch_train_step(lp, mask, s["gradient_accumulation_steps"], s["normalize_constant"])
    assert_close(loss, s["expected_loss"], msg="returned loss")
    assert lp.grad is not None, "did you call loss.backward()?"
    assert_close(lp.grad, s["expected_grad"], msg="gradient w.r.t. policy_log_probs")
    assert int(meta["num_response_tokens"]) == s["expected_num_response_tokens"]
