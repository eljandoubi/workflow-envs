"""Pendulum numerics behind the solver environment."""

import math
import random
from functools import cache

REF_STEP = 0.01
CAND_STEP = 0.05
PERIODS = 10
TOL_ERROR = 0.1
TOL_DRIFT = 0.05
NAIVE_TOL = 1e-3
VISIBLE = (2.0, 0.05, 0.0, 1.0, 400)


def rk4_step(x, v, h, w2):
    k1x, k1v = v, -w2 * math.sin(x)
    k2x, k2v = v + 0.5 * h * k1v, -w2 * math.sin(x + 0.5 * h * k1x)
    k3x, k3v = v + 0.5 * h * k2v, -w2 * math.sin(x + 0.5 * h * k2x)
    k4x, k4v = v + h * k3v, -w2 * math.sin(x + h * k3x)
    x += h / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
    v += h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return x, v


@cache
def reference(omega, x0, v0, t_end):
    """x'' = -omega^2 sin(x) by RK4 with steps five times finer than the candidates get."""
    n = max(1, math.ceil(omega * t_end / REF_STEP))
    h, w2 = t_end / n, omega * omega
    x, v = x0, v0
    for _ in range(n):
        x, v = rk4_step(x, v, h, w2)
    return x, v


def energy(x, v, omega):
    return 0.5 * v * v + omega * omega * (1.0 - math.cos(x))


def errors(omega, x0, v0, t_end, got):
    """Phase-space error relative to the start radius, and relative energy drift."""
    want = reference(omega, x0, v0, t_end)
    radius = math.hypot(x0, v0 / omega)
    error = math.hypot(got[0] - want[0], (got[1] - want[1]) / omega) / radius
    start = energy(x0, v0, omega)
    return error, abs(energy(got[0], got[1], omega) - start) / start


def good_enough(omega, x0, v0, t_end, got):
    if not all(math.isfinite(value) for value in got):
        return False
    error, drift = errors(omega, x0, v0, t_end, got)
    return error <= TOL_ERROR and drift <= TOL_DRIFT


def hidden_configs(seed, count):
    """Held-out cases: swings up to 1.2 rad, ten periods, about 20 steps per radian of phase."""
    rng = random.Random(seed)
    cases = []
    for _ in range(count):
        omega = rng.uniform(1.0, 3.0)
        x0, v0 = rng.uniform(0.3, 1.2), rng.uniform(-1.0, 1.0)
        t_end = PERIODS * 2 * math.pi / omega
        cases.append((omega, x0, v0, t_end, math.ceil(omega * t_end / CAND_STEP)))
    return cases


EXPECTED = reference(*VISIBLE[:4])


def close(got, want):
    return all(abs(a - b) <= NAIVE_TOL for a, b in zip(got, want))


def naive_check(source):
    """The first-draft verifier: run the code in this process, compare one visible example."""
    scope = {"__name__": "candidate"}
    exec(source, scope)
    got = scope["integrate"](*VISIBLE)
    return close(got, EXPECTED)
