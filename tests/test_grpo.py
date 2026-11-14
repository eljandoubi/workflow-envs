import pytest

from workflow_envs import core, grpo, harness


def two_strategies(honest, cheat):
    return {
        "good": {"kind": "honest", "rewards": [honest], "valid": [1.0]},
        "cheat": {"kind": "exploit", "rewards": [cheat], "valid": [0.0]},
    }


def test_softmax_is_a_distribution_and_uniform_for_equal_logits():
    assert sum(grpo.softmax([0.3, -2.0, 5.0])) == pytest.approx(1.0)
    assert grpo.softmax([1.0, 1.0, 1.0]) == pytest.approx([1 / 3] * 3)


def test_equal_rewards_leave_the_policy_where_it_started():
    probs = grpo.train(two_strategies(0.5, 0.5), seed=1)
    assert probs["good"] == pytest.approx(0.5)
    assert probs["cheat"] == pytest.approx(0.5)


def test_the_better_paid_strategy_wins():
    assert grpo.train(two_strategies(0.4, 0.9), seed=1)["cheat"] > 0.9
    assert grpo.train(two_strategies(0.9, 0.4), seed=1)["good"] > 0.9


def test_a_small_gap_is_enough_because_advantages_are_normalised():
    assert grpo.train(two_strategies(0.94, 0.96), seed=2)["cheat"] > 0.9


def test_training_is_repeatable():
    first = grpo.train(two_strategies(0.4, 0.9), seed=3)
    assert grpo.train(two_strategies(0.4, 0.9), seed=3) == first


def test_describe_weights_each_kind_by_its_probability():
    table = two_strategies(0.5, 0.9)
    out = grpo.describe(table, [{"good": 0.25, "cheat": 0.75}])
    assert out["honest"] == pytest.approx(0.25)
    assert out["exploit"] == pytest.approx(0.75)
    assert out["reward"] == pytest.approx(0.25 * 0.5 + 0.75 * 0.9)
    assert out["valid"] == pytest.approx(0.25)


def test_a_naive_verifier_trains_the_cheat_and_a_hardened_one_trains_the_work(rows):
    results = {(r["env"], r["verifier"], r["stage"]): r for r in grpo.study(rows, runs=5)}
    for env in ("ledger", "tickets", "solver"):
        naive = results[(env, "naive", "trained")]
        hardened = results[(env, "hardened", "trained")]
        assert naive["exploit"] > 0.9
        assert naive["valid"] < 0.1
        assert hardened["honest"] > 0.9
        assert hardened["valid"] > 0.9
        assert naive["reward"] > hardened["reward"]


def test_without_a_step_cost_a_naive_verifier_trains_only_what_it_accepts(monkeypatch):
    monkeypatch.setattr(core, "STEP_COST", 0.0)
    rows = harness.evaluate(range(2))
    results = {(r["env"], r["verifier"], r["stage"]): r for r in grpo.study(rows, runs=20)}
    # share of exploits among the strategies the naive verifier accepts: 2 of 4, 2 of 3, 3 of 7
    for env, accepted_share in [("ledger", 2 / 4), ("tickets", 2 / 3), ("solver", 3 / 7)]:
        naive = results[(env, "naive", "trained")]
        assert naive["exploit"] == pytest.approx(accepted_share, abs=0.1)
        assert naive["valid"] > 0.25
        assert results[(env, "hardened", "trained")]["honest"] > 0.9
