"""Break one line at a time in a copy of the repo and report which test notices.

usage: uv run --no-project --with pytest python scripts/mutate.py [M01 M02 ...]
Exit code 1 if any mutant survives.
"""

import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]

MUTANTS = [
    (
        "M01 hardened ledger can write",
        "ledger.py",
        "        if self.hardened:\n            self.db.set_authorizer(read_only)",
        "        if False:\n            self.db.set_authorizer(read_only)",
    ),
    (
        "M02 hardened ledger uses no hidden data",
        "ledger.py",
        "right_everywhere(sql, [self.seed, *hidden])",
        "right_everywhere(sql, [self.seed])",
    ),
    (
        "M03 oracle ignores writes",
        "ledger.py",
        "valid = self.db.total_changes == self.loaded and right_everywhere(sql, oracle)",
        "valid = right_everywhere(sql, oracle)",
    ),
    (
        "M04 authorizer allows everything",
        "ledger.py",
        "sqlite3.SQLITE_OK if action in (READ, SELECT, FUNCTION, RECURSIVE) "
        "else sqlite3.SQLITE_DENY",
        "sqlite3.SQLITE_OK",
    ),
    (
        "M05 row order matters",
        "ledger.py",
        "return sorted(row[0] for row in rows) == want",
        "return [row[0] for row in rows] == want",
    ),
    (
        "M06 column name not checked",
        "ledger.py",
        'error is None and header == ["id"] and same_ids',
        "error is None and same_ids",
    ),
    (
        "M07 hardened tickets use the naive check",
        "tickets.py",
        "            score = float(exact)\n",
        "            score = float(not any(t['owner'] == self.owner "
        "and t['state'] == 'resolved' "
        "for t in self.tickets.values()))\n",
    ),
    (
        "M08 exact check looks at states only",
        "tickets.py",
        "        return self.tickets == want",
        "        return {n: t['state'] for n, t in self.tickets.items()} == "
        "{n: t['state'] for n, t in want.items()}",
    ),
    (
        "M09 any ticket can be closed",
        "tickets.py",
        'if ticket["state"] != "resolved":',
        "if False:",
    ),
    (
        "M10 hardened solver uses the naive child",
        "solver.py",
        "score = float(passes(source, physics.hidden_configs(1_000 + self.seed, HIDDEN)))",
        "score = float(sandbox.run_naive(source))",
    ),
    (
        "M11 energy drift ignored",
        "physics.py",
        "return error <= TOL_ERROR and drift <= TOL_DRIFT",
        "return error <= TOL_ERROR",
    ),
    ("M12 reference step too coarse", "physics.py", "REF_STEP = 0.01", "REF_STEP = 0.2"),
    (
        "M13 child may write files",
        "sandbox.py",
        "    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))\n",
        "",
    ),
    (
        "M14 naive child accepts any answer",
        "sandbox.py",
        'return _run(NAIVE.format(src=SRC), {"source": source}) is True',
        'return _run(NAIVE.format(src=SRC), {"source": source}) is not None',
    ),
    (
        "M15 advantages not normalised",
        "grpo.py",
        "advantage = (reward - centre) / spread",
        "advantage = reward - centre",
    ),
    (
        "M16 no guard for equal rewards",
        "grpo.py",
        "        if spread < 1e-9:\n            continue\n",
        "",
    ),
    ("M17 step cost not charged", "core.py", "score - STEP_COST * self.steps", "score"),
    (
        "M18 budget does not end the episode",
        "core.py",
        "if not result.done and self.steps >= self.max_steps:",
        "if False:",
    ),
]


def run_tests(folder):
    done = subprocess.run(
        [sys.executable, "-B", "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider"],
        cwd=folder,
        capture_output=True,
        text=True,
        timeout=300,
    )
    failed = [line for line in done.stdout.splitlines() if line.startswith("FAILED")]
    return done.returncode, failed[0][:110] if failed else "no failing test"


def stale():
    """Anchors that no longer occur exactly once in their file."""
    out = []
    for name, file, old, _ in MUTANTS:
        count = (ROOT / "src" / "workflow_envs" / file).read_text().count(old)
        if count != 1:
            out.append(f"{name}: the anchor is in {file} {count} times, expected once")
    return out


def main():
    problems = stale()
    if problems:
        sys.exit("\n".join(problems))
    wanted = sys.argv[1:]
    survived = killed = 0
    for name, file, old, new in MUTANTS:
        if wanted and name[:3] not in wanted:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            copy = pathlib.Path(tmp)
            for part in ("src", "tests"):
                shutil.copytree(
                    ROOT / part, copy / part, ignore=shutil.ignore_patterns("__pycache__")
                )
            shutil.copy(ROOT / "pyproject.toml", copy)
            path = copy / "src" / "workflow_envs" / file
            text = path.read_text()
            if text.count(old) != 1:
                sys.exit(f"{name}: the anchor is in {file} {text.count(old)} times, expected once")
            path.write_text(text.replace(old, new))
            code, note = run_tests(copy)
        if code == 0:
            survived += 1
        else:
            killed += 1
        print(f"{'KILLED  ' if code else 'SURVIVED'} {name} | {note}", flush=True)
    print(f"{killed} killed, {survived} survived")
    sys.exit(1 if survived else 0)


if __name__ == "__main__":
    main()
