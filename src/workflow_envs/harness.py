"""Play every policy against every environment and verifier, then summarise."""

import json
import os

from .core import run_episode
from .ledger import Ledger
from .policies import POLICIES
from .solver import Solver
from .tickets import Tickets

ENVS = {"ledger": Ledger, "tickets": Tickets, "solver": Solver}


def evaluate(seeds=range(5)):
    rows = []
    for policy in POLICIES:
        for hardened in (False, True):
            for seed in seeds:
                result = run_episode(ENVS[policy.env](seed=seed, hardened=hardened), policy.play)
                rows.append(
                    {
                        "env": policy.env,
                        "policy": policy.name,
                        "kind": policy.kind,
                        "verifier": "hardened" if hardened else "naive",
                        "seed": seed,
                        "reward": round(result.reward, 4),
                        "score": result.info["score"],
                        "valid": result.info["valid"],
                        "steps": result.info["steps"],
                    }
                )
    return rows


def save(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as out:
        for row in rows:
            out.write(json.dumps(row) + "\n")


def load(path):
    with open(path) as source:
        return [json.loads(line) for line in source]


def mean(values):
    values = list(values)
    return sum(values) / len(values)


def summarise(rows):
    """One entry per policy: share of seeds accepted and share truly valid, per verifier."""
    out = {}
    for row in rows:
        entry = out.setdefault((row["env"], row["policy"]), {"kind": row["kind"]})
        entry.setdefault(row["verifier"], []).append(row)
    summary = []
    for (env, policy), entry in out.items():
        line = {"env": env, "policy": policy, "kind": entry["kind"]}
        for verifier in ("naive", "hardened"):
            line[verifier + "_accepted"] = mean(r["score"] for r in entry[verifier])
            line[verifier + "_valid"] = mean(1.0 if r["valid"] else 0.0 for r in entry[verifier])
            line[verifier + "_steps"] = mean(r["steps"] for r in entry[verifier])
        summary.append(line)
    return summary


def table(rows):
    head = (
        "| env | policy | kind | naive accepts | naive valid "
        "| hardened accepts | hardened valid | steps |"
    )
    lines = [head, "|---|---|---|---|---|---|---|---|"]
    for s in summarise(rows):
        lines.append(
            f"| {s['env']} | {s['policy']} | {s['kind']} "
            f"| {s['naive_accepted']:.2f} | {s['naive_valid']:.2f} "
            f"| {s['hardened_accepted']:.2f} | {s['hardened_valid']:.2f} | {s['naive_steps']:.1f} |"
        )
    return "\n".join(lines)


def accepted_but_invalid(rows, verifier):
    """Episodes the verifier accepted although the strict check says the work is not valid."""
    return sum(
        1 for r in rows if r["verifier"] == verifier and r["score"] == 1.0 and not r["valid"]
    )
