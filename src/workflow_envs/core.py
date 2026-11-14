"""Step result, base environment and episode runner."""

from dataclasses import dataclass, field

STEP_COST = 0.02


@dataclass
class StepResult:
    observation: str
    reward: float = 0.0
    done: bool = False
    info: dict = field(default_factory=dict)


class Env:
    """Text actions in, text observations out.

    `hardened` picks the verifier: False is the naive check a first draft would
    ship, True is the one that holds against the policies in policies.py.
    The reward is the verifier score minus STEP_COST per action, so a shorter
    route to the same score pays more. `info["valid"]` comes from a stricter
    check that both verifiers are measured against; agents never see it.
    """

    name = ""
    max_steps = 8

    def __init__(self, seed=0, hardened=False):
        self.seed = seed
        self.hardened = hardened
        self.steps = 0
        self.done = False

    def reset(self):
        self.steps = 0
        self.done = False
        return self.start()

    def step(self, action):
        if self.done:
            raise RuntimeError("episode is over, call reset()")
        self.steps += 1
        result = self.act(action)
        if not result.done and self.steps >= self.max_steps:
            result = self.finish(0.0, False, result.observation + "\nstep budget used up")
        return result

    def finish(self, score, valid, observation):
        self.done = True
        info = {"score": score, "valid": valid, "steps": self.steps}
        return StepResult(observation, score - STEP_COST * self.steps, True, info)

    def start(self):
        raise NotImplementedError

    def act(self, action):
        raise NotImplementedError


def run_episode(env, play):
    """Run one policy, a generator that yields actions and is sent observations."""
    moves = play(env.reset())
    try:
        action = next(moves)
        while True:
            result = env.step(action)
            if result.done:
                return result
            action = moves.send(result.observation)
    except StopIteration:
        return env.finish(0.0, False, "the policy stopped without finishing")
