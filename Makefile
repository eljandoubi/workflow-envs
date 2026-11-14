PY ?= python3
export PYTHONPATH := src

test:
	uv run --no-project --with pytest python -m pytest -q

results:
	mkdir -p results
	$(PY) -m workflow_envs evaluate --seeds 5 --out results/episodes.jsonl > results/episodes.md
	$(PY) -m workflow_envs train --episodes results/episodes.jsonl > results/training.md

.PHONY: test results
