"""python -m workflow_envs evaluate | train"""

import argparse

from . import grpo, harness


def main():
    parser = argparse.ArgumentParser(prog="workflow_envs")
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate", help="play every policy against both verifiers")
    evaluate.add_argument("--seeds", type=int, default=5)
    evaluate.add_argument("--out", default="results/episodes.jsonl")
    train = commands.add_parser("train", help="group-relative update over the scripted strategies")
    train.add_argument("--episodes", default="results/episodes.jsonl")
    train.add_argument("--runs", type=int, default=20)
    args = parser.parse_args()
    if args.command == "evaluate":
        rows = harness.evaluate(range(args.seeds))
        harness.save(rows, args.out)
        print(harness.table(rows))
    else:
        print(grpo.table_text(grpo.study(harness.load(args.episodes), runs=args.runs)))


if __name__ == "__main__":
    main()
