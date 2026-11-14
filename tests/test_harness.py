import os
import subprocess
import sys
from pathlib import Path

from workflow_envs import harness
from workflow_envs.policies import POLICIES

SRC = str(Path(__file__).resolve().parents[1] / "src")


def of_kind(rows, kind, verifier):
    return [row for row in rows if row["kind"] == kind and row["verifier"] == verifier]


def test_every_policy_is_played_against_both_verifiers(rows):
    assert len(rows) == len(POLICIES) * 2 * 2
    assert {row["verifier"] for row in rows} == {"naive", "hardened"}
    assert {row["seed"] for row in rows} == {0, 1}


def test_honest_policies_pass_and_are_valid_under_both_verifiers(rows):
    for verifier in ("naive", "hardened"):
        for row in of_kind(rows, "honest", verifier):
            assert row["score"] == 1.0
            assert row["valid"]


def test_exploits_pass_the_naive_verifiers_and_never_the_hardened_ones(rows):
    assert all(r["score"] == 1.0 and not r["valid"] for r in of_kind(rows, "exploit", "naive"))
    assert all(r["score"] == 0.0 for r in of_kind(rows, "exploit", "hardened"))


def test_wrong_policies_are_never_valid(rows):
    assert not any(row["valid"] for row in rows if row["kind"] == "wrong")


def test_nothing_the_hardened_verifiers_accept_is_invalid(rows):
    assert harness.accepted_but_invalid(rows, "hardened") == 0
    assert harness.accepted_but_invalid(rows, "naive") > 0


def test_exploits_cost_fewer_steps_than_honest_work(rows):
    for env in ("ledger", "tickets", "solver"):
        honest = [r["steps"] for r in rows if r["env"] == env and r["kind"] == "honest"]
        cheapest = [r["steps"] for r in rows if r["env"] == env and r["kind"] == "exploit"]
        assert min(cheapest) < min(honest)


def test_the_table_has_a_line_per_policy(rows):
    assert len(harness.summarise(rows)) == len(POLICIES)
    assert len(harness.table(rows).splitlines()) == len(POLICIES) + 2


def test_save_and_load_round_trip(rows, tmp_path):
    path = str(tmp_path / "out" / "episodes.jsonl")
    harness.save(rows, path)
    assert harness.load(path) == rows


def test_the_train_command_reads_saved_episodes(rows, tmp_path):
    path = str(tmp_path / "episodes.jsonl")
    harness.save(rows, path)
    done = subprocess.run(
        [sys.executable, "-m", "workflow_envs", "train", "--episodes", path, "--runs", "2"],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": SRC},
    )
    lines = done.stdout.splitlines()
    assert lines[0].startswith("| env | verifier | stage")
    assert len(lines) == 2 + 3 * 2 * 2
