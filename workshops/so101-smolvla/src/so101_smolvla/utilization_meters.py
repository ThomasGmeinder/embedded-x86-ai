"""Open and close the utilization and power table from a notebook or another program. Standard library only.

    from so101_smolvla.utilization_meters import open_meters, close_meters
    facts = open_meters()     # starts the window program if none runs; returns the facts of its "shown" line
    close_meters()            # ends it, from any process

The window program (utilization_window, through scripts/run_utilization_meters.sh) runs on the machine of
the caller and draws on the display the caller's process has. "Is it running" has one authority: the lock
file in the user's runtime directory, held with flock by the program while it lives, with its facts inside.
A program that crashed releases the lock by itself.
"""

from __future__ import annotations

import fcntl
import json
import os
import select
import signal
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "scripts" / "run_utilization_meters.sh"
LOCK_NAME = "so101-utilization-meters.lock"
LOG_NAME = "so101-utilization-meters.log"


def runtime_dir() -> Path:
    """The user's runtime directory (XDG_RUNTIME_DIR), or the temporary directory with the user's id."""
    path = os.environ.get("XDG_RUNTIME_DIR")
    if path and Path(path).is_dir():
        return Path(path)
    fallback = Path(f"/tmp/so101-utilization-meters-{os.getuid()}")
    fallback.mkdir(mode=0o700, exist_ok=True)
    return fallback


class HeldLock:
    """The lock file as the window program holds it."""

    def __init__(self, path: Path, fd: int) -> None:
        self.path, self._fd = path, fd

    def write(self, facts: dict) -> None:
        os.ftruncate(self._fd, 0)
        os.lseek(self._fd, 0, os.SEEK_SET)
        os.write(self._fd, (json.dumps(facts) + "\n").encode("utf-8"))
        os.fsync(self._fd)

    def release(self) -> None:
        os.ftruncate(self._fd, 0)
        fcntl.flock(self._fd, fcntl.LOCK_UN)
        os.close(self._fd)


def hold_lock(path: Path) -> HeldLock | None:
    """Take the lock, or None when another process holds it."""
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    return HeldLock(path, fd)


def running_facts(lock_path: Path | None = None) -> dict | None:
    """The facts of the running window program, or None when none holds the lock."""
    path = lock_path or runtime_dir() / LOCK_NAME
    if not path.exists():
        return None
    fd = os.open(path, os.O_RDONLY)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            text = path.read_text(encoding="utf-8").strip()
            # The holder writes its facts right after its first frame; before that the file is empty.
            return json.loads(text) if text else {"starting": True}
        fcntl.flock(fd, fcntl.LOCK_UN)
        return None
    finally:
        os.close(fd)


def _parse_shown(line: str) -> dict:
    facts = dict(part.split("=", 1) for part in line.split()[1:])
    for key in ("pid", "window"):
        facts[key] = int(facts[key])
    return facts


def open_meters(place: tuple[int, int] | None = None, launcher: Path = LAUNCHER) -> dict:
    """Start the table if it is not running and return its facts (pid, display, window, renderer, host, log, lock).

    Blocks until the program prints its "shown" line. A program that ends first raises RuntimeError with
    its output (for example the no-display error).
    """
    facts = running_facts()
    if facts:
        return facts
    command = [str(launcher)]
    if place is not None:
        command += ["--place", f"{place[0]},{place[1]}"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                               start_new_session=True)
    lines = []
    for line in process.stdout:
        if line.startswith("shown "):
            process.stdout.close()
            return _parse_shown(line)
        lines.append(line)
    code = process.wait()
    raise RuntimeError(f"the window program ended with status {code} before it was shown:\n{''.join(lines)}")


def close_meters(timeout_s: float = 10.0) -> bool:
    """End the running table; True when one ran. Waits for its exit as an event (a process descriptor)."""
    facts = running_facts()
    if not facts or "pid" not in facts:
        return False
    pid = int(facts["pid"])
    try:
        pidfd = os.pidfd_open(pid)
    except ProcessLookupError:
        return False
    try:
        os.kill(pid, signal.SIGTERM)
        ready, _, _ = select.select([pidfd], [], [], timeout_s)
        if not ready:
            raise TimeoutError(f"the window program (pid {pid}) did not end within {timeout_s} s of SIGTERM")
    finally:
        os.close(pidfd)
    try:
        os.waitpid(pid, 0)  # reaps it when it is this process's child
    except ChildProcessError:
        pass
    return True


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="open or close the utilization and power table")
    parser.add_argument("action", choices=("open", "close", "status"))
    args = parser.parse_args(argv)
    if args.action == "open":
        print(json.dumps(open_meters()))
    elif args.action == "close":
        print("closed" if close_meters() else "nothing was running")
    else:
        print(json.dumps(running_facts()))


if __name__ == "__main__":
    main()
