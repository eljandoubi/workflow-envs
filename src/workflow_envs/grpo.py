"""Group-relative policy gradient over a few scripted strategies.

One action is a whole strategy. Rewards come from recorded episodes (each
environment is deterministic for a given seed), so training costs nothing.
The policy is a tabular softmax, not a language model.
"""

import math
import random

from .harness import mean

ITERATIONS = 200
GROUP = 16
LEARNING_RATE = 0.5
KINDS = ("honest", "wrong", "exploit")


def softmax(logits):
    top = max(logits)
    weights = [math.exp(value - top) for value in logits]
    total = sum(weights)
    return [weight / total for weight in weights]


def reward_table(rows, env, verifier):
    """policy -> kind, rewards and valid flags, one entry per task seed."""
    table = {}
    for row in rows:
        if row["env"] == env and row["verifier"] == verifier:
            entry = table.setdefault(
                row["policy"], {"kind": row["kind"], "rewards": [], "valid": []}
            )
            entry["rewards"].append(row["reward"])
            entry["valid"].append(1.0 if row["valid"] else 0.0)
    return table


def train(table, seed=0, iterations=ITERATIONS, group=GROUP, lr=LEARNING_RATE):
    """Probability of each strategy after training from a uniform start."""
    names = list(table)
    rng = random.Random(seed)
    logits = [0.0] * len(names)
    for _ in range(iterations):
        probs = softmax(logits)
        picks = rng.choices(range(len(names)), probs, k=group)
        rewards = [rng.choice(table[names[k]]["rewards"]) for k in picks]
        centre = sum(rewards) / group
        spread = math.sqrt(sum((r - centre) ** 2 for r in rewards) / group)
        if spread < 1e-9:
            continue
        grad = [0.0] * len(names)
        for k, reward in zip(picks, rewards):
            advantage = (reward - centre) / spread
            for j in range(len(names)):
                grad[j] += advantage * ((1.0 if j == k else 0.0) - probs[j])
        logits = [value + lr * g / group for value, g in zip(logits, grad)]
    return dict(zip(names, softmax(logits)))


def describe(table, probs_list):
    """Mean probability per kind, expected reward and expected valid rate over several policies."""
    out = {kind: 0.0 for kind in KINDS}
    out["reward"] = out["valid"] = 0.0
    for probs in probs_list:
        for name, p in probs.items():
            entry = table[name]
            out[entry["kind"]] += p / len(probs_list)
            out["reward"] += p * mean(entry["rewards"]) / len(probs_list)
            out["valid"] += p * mean(entry["valid"]) / len(probs_list)
    return out


def study(rows, runs=20, **options):
    """Uniform start and trained policy per environment and verifier, mean over training seeds."""
    results = []
    for env in dict.fromkeys(row["env"] for row in rows):
        for verifier in ("naive", "hardened"):
            table = reward_table(rows, env, verifier)
            uniform = {name: 1 / len(table) for name in table}
            trained = [train(table, seed=seed, **options) for seed in range(runs)]
            results.append(
                {"env": env, "verifier": verifier, "stage": "start", **describe(table, [uniform])}
            )
            results.append(
                {"env": env, "verifier": verifier, "stage": "trained", **describe(table, trained)}
            )
    return results


def table_text(results):
    lines = [
        "| env | verifier | stage | honest | wrong | exploit | reward | valid |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['env']} | {r['verifier']} | {r['stage']} | {r['honest']:.2f} | {r['wrong']:.2f} "
            f"| {r['exploit']:.2f} | {r['reward']:.2f} | {r['valid']:.2f} |"
        )
    return "\n".join(lines)
