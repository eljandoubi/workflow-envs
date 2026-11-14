"""Run candidate code in a child process with CPU, file and time limits.

This keeps slow or sloppy code from stalling a run. It is not a security
boundary: the child can still read files and open sockets. POSIX only.
"""

import json
import os
import resource
import subprocess
import sys
import tempfile

CPU_SECONDS = 10
WALL_SECONDS = 20
SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# a shared libpython can need this to start; nothing else reaches the child
LOADER_ENV = {name: os.environ[name] for name in ("LD_LIBRARY_PATH",) if name in os.environ}

PLAIN = """
import json, sys
job = json.load(sys.stdin)
scope = {"__name__": "candidate"}
exec(job["source"], scope)
print(json.dumps([[float(c) for c in scope["integrate"](*case)] for case in job["configs"]]))
"""

NAIVE = """
import json, sys
sys.path.insert(0, {src!r})
from workflow_envs import physics
job = json.load(sys.stdin)
print(json.dumps(physics.naive_check(job["source"])))
"""


def _limits():
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    if sys.platform == "linux":
        resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))


def _run(code, job):
    """Last line of the child's stdout as JSON, or None if it failed or timed out."""
    with tempfile.TemporaryDirectory() as cwd:
        try:
            done = subprocess.run(
                [sys.executable, "-I", "-B", "-c", code],
                input=json.dumps(job),
                capture_output=True,
                text=True,
                timeout=WALL_SECONDS,
                cwd=cwd,
                env=LOADER_ENV,
                preexec_fn=_limits,
            )
        except subprocess.TimeoutExpired:
            return None
    if done.returncode != 0:
        return None
    try:
        return json.loads(done.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        return None


def run_candidate(source, configs):
    """integrate(*config) for each config, as (x, v) pairs, or None if the code failed.

    The child holds no reference values. Its answers come back as plain numbers
    and are judged by the caller, so forging the output line only means
    claiming numbers.
    """
    out = _run(PLAIN, {"source": source, "configs": [list(case) for case in configs]})
    if (
        isinstance(out, list)
        and len(out) == len(configs)
        and all(isinstance(row, list) and len(row) == 2 for row in out)
    ):
        return [tuple(row) for row in out]
    return None


def run_naive(source):
    """The first-draft check, run in a child that also holds the expected answer."""
    return _run(NAIVE.format(src=SRC), {"source": source}) is True
