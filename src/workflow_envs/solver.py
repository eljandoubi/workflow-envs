"""Code environment: write a pendulum integrator, graded by a verifier."""

from . import physics, sandbox
from .core import Env, StepResult

HIDDEN = 4
ORACLE = 10

TASK = (
    "Write Python source that defines integrate(omega, x0, v0, t_end, steps) for the pendulum "
    "x'' = -omega**2 * sin(x). It takes `steps` fixed steps and returns (x, v) at t_end. "
    "Example: integrate(2.0, 0.05, 0.0, 1.0, 400) is about ({x:.6f}, {v:.6f}). "
    "It is graded on held-out cases that include long runs of about ten periods, "
    "where the energy 0.5*v**2 + omega**2*(1 - cos(x)) has to stay close to its starting value. "
    "Send `test: <code>` to run the example and `submit: <code>` to finish."
)


def passes(source, cases):
    answers = sandbox.run_candidate(source, cases)
    if answers is None:
        return False
    return all(physics.good_enough(*case[:4], got) for case, got in zip(cases, answers))


class Solver(Env):
    name = "solver"
    max_steps = 4

    def start(self):
        x, v = physics.EXPECTED
        return TASK.format(x=x, v=v)

    def act(self, action):
        verb, _, body = action.partition(":")
        verb, body = verb.strip().lower(), body.strip()
        if verb == "test":
            return StepResult(self.test(body))
        if verb == "submit":
            return self.grade(body)
        return StepResult("unknown action: use `test: <code>` or `submit: <code>`")

    def test(self, source):
        answers = sandbox.run_candidate(source, [physics.VISIBLE])
        if answers is None:
            return "the code failed: error, timeout or no answer"
        return f"integrate{physics.VISIBLE} -> ({answers[0][0]:.6f}, {answers[0][1]:.6f})"

    def grade(self, source):
        if self.hardened:
            score = float(passes(source, physics.hidden_configs(1_000 + self.seed, HIDDEN)))
        else:
            score = float(sandbox.run_naive(source))
        valid = passes(source, physics.hidden_configs(5_000 + self.seed, ORACLE))
        return self.finish(score, valid, "accepted" if score else "rejected")
