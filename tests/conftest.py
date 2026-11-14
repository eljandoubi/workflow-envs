import pytest

from workflow_envs.core import run_episode
from workflow_envs.harness import ENVS, evaluate
from workflow_envs.policies import POLICIES


@pytest.fixture
def play():
    def run(env, name, hardened, seed=0):
        policy = next(p for p in POLICIES if p.env == env and p.name == name)
        return run_episode(ENVS[env](seed=seed, hardened=hardened), policy.play)

    return run


@pytest.fixture(scope="session")
def rows():
    return evaluate(range(2))
