import pytest

from workflow_envs.core import STEP_COST, Env, StepResult, run_episode


class Counter(Env):
    max_steps = 3

    def start(self):
        return "go"

    def act(self, action):
        if action == "stop":
            return self.finish(1.0, True, "ok")
        return StepResult("again")


def script(*actions):
    def play(_task):
        for action in actions:  # noqa: UP028 (yield from would break send())
            yield action

    return play


def test_reward_is_the_score_minus_a_cost_per_action():
    result = run_episode(Counter(), script("a", "b", "stop"))
    assert result.done
    assert result.reward == pytest.approx(1.0 - 3 * STEP_COST)
    assert result.info == {"score": 1.0, "valid": True, "steps": 3}


def test_a_shorter_route_to_the_same_score_pays_more():
    short = run_episode(Counter(), script("stop"))
    long = run_episode(Counter(), script("a", "stop"))
    assert short.reward > long.reward


def test_the_step_budget_ends_the_episode_with_score_zero():
    result = run_episode(Counter(), script("a", "a", "a", "a"))
    assert result.done
    assert result.info["score"] == 0.0
    assert result.info["steps"] == 3
    assert "budget" in result.observation


def test_a_policy_that_stops_early_scores_zero():
    result = run_episode(Counter(), script())
    assert result.done
    assert result.info["score"] == 0.0


def test_stepping_after_the_end_is_an_error():
    env = Counter()
    env.reset()
    env.step("stop")
    with pytest.raises(RuntimeError):
        env.step("stop")
