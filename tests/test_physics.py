import math

import pytest

from workflow_envs import physics
from workflow_envs.policies import EULER, RK4, SMALL_ANGLE, VERLET

CASES = physics.hidden_configs(0, 10)


def load(source):
    scope = {}
    exec(source, scope)
    return scope["integrate"]


def test_reference_agrees_with_a_four_times_finer_run():
    omega, x0, v0, t_end, _ = CASES[0]
    n = math.ceil(omega * t_end / (physics.REF_STEP / 4))
    h, w2 = t_end / n, omega * omega
    x, v = x0, v0
    for _ in range(n):
        x, v = physics.rk4_step(x, v, h, w2)
    want = physics.reference(omega, x0, v0, t_end)
    assert abs(x - want[0]) < 1e-8
    assert abs(v - want[1]) < 1e-8


def test_reference_keeps_the_energy():
    for omega, x0, v0, t_end, _ in CASES[:3]:
        x, v = physics.reference(omega, x0, v0, t_end)
        start = physics.energy(x0, v0, omega)
        assert abs(physics.energy(x, v, omega) - start) / start < 1e-6


def test_hidden_cases_are_repeatable_and_stay_below_the_separatrix():
    assert physics.hidden_configs(3, 4) == physics.hidden_configs(3, 4)
    assert physics.hidden_configs(3, 4) != physics.hidden_configs(4, 4)
    for omega, x0, v0, t_end, steps in CASES:
        assert physics.energy(x0, v0, omega) < 2 * omega**2
        assert steps == math.ceil(omega * t_end / physics.CAND_STEP)


@pytest.mark.parametrize("source", [VERLET, RK4])
def test_good_integrators_pass_with_room_to_spare(source):
    integrate = load(source)
    for case in CASES:
        error, drift = physics.errors(*case[:4], integrate(*case))
        assert error < physics.TOL_ERROR / 3
        assert drift < physics.TOL_DRIFT / 3


def test_euler_gains_energy_and_fails_every_case():
    integrate = load(EULER)
    assert not any(physics.good_enough(*case[:4], integrate(*case)) for case in CASES)
    assert physics.errors(*CASES[0][:4], integrate(*CASES[0]))[1] > 1.0


def test_the_small_angle_formula_fails_every_case():
    integrate = load(SMALL_ANGLE)
    assert not any(physics.good_enough(*case[:4], integrate(*case)) for case in CASES)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_answers_that_are_not_finite_fail(bad):
    assert not physics.good_enough(*CASES[0][:4], (bad, 0.0))


def test_the_reference_answer_passes_and_a_shifted_one_fails():
    omega, x0, v0, t_end, _ = CASES[0]
    x, v = physics.reference(omega, x0, v0, t_end)
    assert physics.good_enough(omega, x0, v0, t_end, (x, v))
    assert not physics.good_enough(omega, x0, v0, t_end, (x + 0.5 * math.hypot(x0, v0 / omega), v))


def test_a_small_radius_error_can_still_break_the_energy_check():
    omega, x0, v0, t_end, _ = CASES[0]
    x, v = physics.reference(omega, x0, v0, t_end)
    got = (x * 1.07, v * 1.07)
    error, drift = physics.errors(omega, x0, v0, t_end, got)
    assert error <= physics.TOL_ERROR
    assert drift > physics.TOL_DRIFT
    assert not physics.good_enough(omega, x0, v0, t_end, got)


def test_the_naive_check_looks_at_one_short_example_only():
    assert physics.naive_check(VERLET)
    assert physics.naive_check(EULER)
    assert physics.naive_check(SMALL_ANGLE)
    assert not physics.naive_check(
        "def integrate(omega, x0, v0, t_end, steps):\n    return 0.0, 0.0"
    )
