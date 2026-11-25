# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The robot stack ``run_pipeline.sh`` and the digital-twin notebook start.

``python -m so101_sim.stack`` replaces itself with one ``ros2 launch`` of
``launch/strix_vla.launch.py``. ``python -m so101_sim.stack serve`` starts the
whole stack and is what ``run_pipeline.sh`` execs: the camera launch, the arm
launch, perception, the OpenCV console, and control, each on the core named in
``cores`` of the pipeline config. The digital-twin notebook calls
:func:`start_digital_twin`, which starts the same two launches.

This process stops them, in the same order, when any of these happens:

- the OpenCV console window is closed, or control exits;
- this process gets SIGINT or SIGTERM (Ctrl-C on ``run_pipeline.sh``);
- the notebook kernel exits (the notebook ended, or Ctrl-C on the Jupyter
  server, which shuts the kernel down). The launches are children of the
  kernel, and each launch gets one SIGINT so the webcam driver is signaled once.
"""

from __future__ import annotations

import argparse
import atexit
import ctypes
import os
import select
import shutil
import signal
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LAUNCH = REPO / "launch" / "strix_vla.launch.py"


def command(*, arm: bool, cameras: bool, digital_twin: bool, dry_run: bool) -> list[str]:
    """The ``python -m so101_sim.stack`` command ``run_pipeline.sh`` runs."""
    return [
        sys.executable, "-m", "so101_sim.stack",
        "--arm", str(arm).lower(),
        "--cameras", str(cameras).lower(),
        "--digital-twin", str(digital_twin).lower(),
        "--dry-run", str(dry_run).lower(),
    ]


def launch_argv(
    *,
    arm: bool,
    cameras: bool,
    digital_twin: bool,
    dry_run: bool,
    config: str | None = None,
) -> list[str]:
    """The launch's command, with the twin, control and perception cores:
    ``ros2 launch``, through :mod:`so101_sim.ros_launch`."""
    from so101_sim.ros_launch import python_args
    from vla_pipeline.utils.config import load_config

    cores = load_config(config).get("cores") or {}
    arguments = [
        f"arm:={str(arm).lower()}",
        f"cameras:={str(cameras).lower()}",
        f"digital_twin:={str(digital_twin).lower()}",
        f"dry_run:={str(dry_run).lower()}",
        f"twin_core:={cores.get('twin', '')}",
        f"control_core:={cores.get('control', '')}",
        f"perception_core:={cores.get('perception', '')}",
    ]
    if config:
        arguments.append(f"config:={config}")
    return [sys.executable, *python_args(LAUNCH, *arguments, options=("--noninteractive",))]


def accept_stop_signals() -> None:
    """Leave a background shell job able to stop on SIGINT.

    ``run_pipeline.sh`` starts this module with ``&``. A non-interactive bash
    does not give that job its own process group, and it inherits SIGINT
    ignored. ``signal.signal`` cannot clear an inherited ignore. The cleanup
    in the script signals the group (``kill -INT -- -$pid``), so this process
    has to lead the group and accept the signal before it becomes the launch.
    """
    if os.getpgrp() != os.getpid():
        os.setsid()
    libc = ctypes.CDLL(None)
    libc.signal(signal.SIGINT, 0)
    libc.signal(signal.SIGQUIT, 0)


def hold_domain_lock() -> None:
    """Hold this domain's twin lock across the exec, until the launch exits.

    The notebook checks the lock instead of waiting to see whether joint states
    show up. The file descriptor has to stay open in the launched process.
    """
    from so101_sim.domain_lock import acquire

    domain_id = int(os.environ.get("ROS_DOMAIN_ID", "0"))
    handle = acquire(domain_id)
    os.set_inheritable(handle.fileno(), True)
    # Referenced until exec. A dropped reference closes the descriptor.
    global _DOMAIN_LOCK
    _DOMAIN_LOCK = handle


def exec_launch(**kwargs) -> None:
    """Replace this process with the launch. The process dies with its parent.

    The parent's exit delivers one SIGINT to this process, which is the launch.
    A signal to the launch's whole group would also hit the webcam, and the
    launch's own forwarded SIGINT then wedges the AMD ISP driver.
    """
    from so101_sim.parent_death import die_with_parent

    die_with_parent(signal.SIGINT)
    if kwargs.get("arm"):
        hold_domain_lock()
    argv = launch_argv(**kwargs)
    accept_stop_signals()
    os.execvp(argv[0], argv)


def child_env(base: dict | None = None) -> dict:
    """The environment a stack process inherits.

    ``FASTRTPS_DEFAULT_PROFILES_FILE`` pointing at the localhost profile turns
    multicast off. With ``FASTDDS_BUILTIN_TRANSPORTS=LARGE_DATA`` the controller
    manager then never receives ``robot_description``, and no joint states are
    published. ``run_pipeline.sh`` does not set that profile.
    """
    env = dict(os.environ if base is None else base)
    env.pop("FASTRTPS_DEFAULT_PROFILES_FILE", None)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def lead(argv: list[str]) -> None:
    """Own the session, die with the parent, then exec ``argv``.

    ``lead --core N -- cmd args`` pins the command with ``taskset``. The parent
    is the stack session. A background job cannot do this with ``preexec_fn``:
    the notebook kernel is threaded, and a pre-exec function there can run on
    the wrong thread.
    """
    from so101_sim.parent_death import die_with_parent

    core = ""
    if argv[:1] == ["--core"]:
        core = argv[1]
        argv = argv[2:]
    if argv[:1] == ["--"]:
        argv = argv[1:]
    if not argv:
        raise SystemExit("usage: python -m so101_sim.stack lead [--core N] -- CMD [ARGS...]")
    die_with_parent(signal.SIGTERM)
    if os.getpgrp() != os.getpid():
        os.setsid()
    if core:
        os.execvp("taskset", ["taskset", "-c", core, *argv])
    os.execvp(argv[0], argv)


class Session:
    """The stack's processes, and the one place that stops them.

    Perception and the console stop first, so a late command cannot arrive
    while control parks the arm. Control is next. Each launch then gets one
    SIGINT to its own process, not to its group.
    """

    def __init__(self) -> None:
        self.launches: list[subprocess.Popen] = []
        self.sides: list[subprocess.Popen] = []
        self.console: subprocess.Popen | None = None
        self.control: subprocess.Popen | None = None
        self._stopped = False
        self._wake_r, self._wake_w = os.pipe()

    def notify(self) -> None:
        """Wake :meth:`wait`. A signal handler may call this."""
        try:
            os.write(self._wake_w, b"\0")
        except OSError:
            pass

    def running(self) -> bool:
        """A process this session started has not exited."""
        procs = [*self.launches, *self.sides]
        if self.control is not None:
            procs.append(self.control)
        return any(proc.poll() is None for proc in procs)

    def add_launch(
        self, argv: list[str], *, env: dict | None = None, stdout=None, pass_fds: tuple[int, ...] = (),
    ) -> subprocess.Popen:
        """One launch. The child makes itself the session leader."""
        proc = _popen(argv, env=child_env(env), stdout=stdout, stdin=subprocess.DEVNULL, pass_fds=pass_fds)
        self.launches.append(proc)
        return proc

    def add_side(
        self, argv: list[str], *, core: str = "", env: dict | None = None, stdout=None,
        pass_fds: tuple[int, ...] = (),
    ) -> subprocess.Popen:
        """Perception or the console, in its own session, on ``core`` when set."""
        proc = _popen(
            _lead_command(argv, core), env=child_env(env), stdout=stdout,
            stdin=subprocess.DEVNULL, pass_fds=pass_fds,
        )
        self.sides.append(proc)
        return proc

    def add_control(
        self, argv: list[str], *, core: str = "", env: dict | None = None, stdout=None,
        pass_fds: tuple[int, ...] = (),
    ) -> subprocess.Popen:
        """Control keeps this process's stdin, so typed commands reach it."""
        proc = _popen(
            _lead_command(argv, core), env=child_env(env), stdout=stdout, stdin=None, pass_fds=pass_fds,
        )
        self.control = proc
        return proc

    def start_launches(
        self, *, arm: bool, cameras: bool, digital_twin: bool, dry_run: bool,
        stdout=None, env: dict | None = None,
    ) -> None:
        """The camera launch, then the arm launch. The same split ``run_pipeline.sh`` uses.

        ros2 launch ends every part of a launch when one part raises, so the
        webcam never shares a launch with the arm.
        """
        if cameras:
            self.add_launch(
                command(arm=False, cameras=True, digital_twin=digital_twin, dry_run=dry_run),
                env=env, stdout=stdout,
            )
        if arm:
            self.add_launch(
                command(arm=True, cameras=False, digital_twin=digital_twin, dry_run=dry_run),
                env=env, stdout=stdout,
            )

    def start_pipeline(self, argv: list[str]) -> None:
        """The processes ``run_pipeline.sh`` used to start from the shell."""
        from vla_pipeline.utils.config import load_config

        cfg = load_config()
        robot = str(cfg["robot"].get("use_ros2", False)).lower() == "true"
        cams = cfg.get("cameras") or {}
        cameras = any(
            str((cams.get(role) or {}).get("use_ros2", False)).lower() == "true"
            for role in ("mount", "arm")
        )
        dry_run = "--dry-run" in argv
        digital_twin = "--digital-twin" in argv and not dry_run and robot
        if robot or cameras:
            if shutil.which("ros2") is None:
                raise RuntimeError("ros2 is not on PATH: source /opt/ros/jazzy/setup.bash first")
            if cameras:
                print("Starting the camera nodes...", flush=True)
            if robot:
                print(f"Starting the arm nodes (digital twin: {str(digital_twin).lower()})...", flush=True)
            self.start_launches(arm=robot, cameras=cameras, digital_twin=digital_twin, dry_run=dry_run)
        cores = cfg.get("cores") or {}
        self.add_side(
            [sys.executable, "-m", "vla_pipeline.perception", *argv],
            core=str(cores.get("perception") or ""),
        )
        if "--digital-twin" in argv and "--no-console" not in argv and "--headless" not in argv:
            self.console = self.add_side(
                [sys.executable, "-m", "vla_pipeline.console_main", *argv],
                core=str(cores.get("twin") or ""),
            )
        self.add_control(
            [sys.executable, "-m", "vla_pipeline.main", "--no-console", *argv],
            core=str(cores.get("control") or ""),
        )

    def wait(self) -> None:
        """Block until the console or control exits, or :meth:`notify` is called.

        The console process exits when its OpenCV window is closed. Control
        exits when its stdin closes or it is asked to stop. One ``select`` on
        those process descriptors and the signal pipe is the wait. A process
        that has already exited is that event.
        """
        watched = [proc for proc in (self.console, self.control) if proc is not None]
        if any(proc.poll() is not None for proc in watched):
            return
        pidfds = []
        try:
            for proc in watched:
                pidfds.append(os.pidfd_open(proc.pid))
            while True:
                try:
                    select.select([self._wake_r, *pidfds], [], [])
                    return
                except InterruptedError:
                    continue
        finally:
            for fd in pidfds:
                os.close(fd)

    def stop(self) -> None:
        """Stop perception and the console, then control, then each launch."""
        if self._stopped:
            return
        self._stopped = True
        for proc in self.sides:
            _signal_group(proc, signal.SIGTERM)
        for proc in self.sides:
            proc.wait()
        if self.control is not None:
            _signal_group(self.control, signal.SIGTERM)
            self.control.wait()
        for proc in self.launches:
            _signal_pid(proc, signal.SIGINT)
        for proc in self.launches:
            proc.wait()

    def return_code(self) -> int:
        """Control's exit code, or 0 when this session is what ended it."""
        if self.control is None or self.control.returncode is None or self.control.returncode < 0:
            return 0
        return self.control.returncode


def _lead_command(argv: list[str], core: str) -> list[str]:
    cmd = [sys.executable, "-m", "so101_sim.stack", "lead"]
    if core:
        cmd.extend(["--core", core])
    return [*cmd, "--", *argv]


def _popen(argv: list[str], *, env: dict, stdout, stdin, pass_fds: tuple[int, ...] = ()) -> subprocess.Popen:
    return subprocess.Popen(
        argv, env=env, stdout=stdout,
        stderr=subprocess.STDOUT if stdout is not None else None,
        stdin=stdin, pass_fds=pass_fds,
    )


def _signal_pid(proc: subprocess.Popen, sig: int) -> None:
    """One signal to ``proc`` itself. Not to its group."""
    if proc.poll() is not None:
        return
    try:
        os.kill(proc.pid, sig)
    except ProcessLookupError:
        pass


def _signal_group(proc: subprocess.Popen, sig: int) -> None:
    """Signal ``proc``'s session, so a child it started (llama-server) gets it too."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        _signal_pid(proc, sig)


_notebook_session: Session | None = None


def start_digital_twin() -> Session:
    """The camera and arm launches of ``./run_pipeline.sh --digital-twin``.

    The notebook kernel calls this. A second call while those launches are
    alive returns the same session. When the kernel exits, :meth:`Session.stop`
    runs: the notebook was closed, or Ctrl-C on the Jupyter server shut the
    kernel down. The log is ``workshop/logs/twin.log``.
    """
    global _notebook_session
    if _notebook_session is not None and _notebook_session.running():
        return _notebook_session
    session = Session()
    log_path = REPO / "workshop" / "logs" / "twin.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "a", encoding="utf-8", buffering=1)
    session.start_launches(
        arm=True, cameras=True, digital_twin=True, dry_run=False, stdout=log,
    )
    atexit.register(session.stop)
    _notebook_session = session
    return session


def serve(argv: list[str]) -> None:
    """Start the stack and stop it when the console closes, control exits, or a signal arrives."""
    session = Session()
    signal.signal(signal.SIGINT, lambda *_: session.notify())
    signal.signal(signal.SIGTERM, lambda *_: session.notify())
    try:
        session.start_pipeline(argv)
        session.wait()
    finally:
        session.stop()
    raise SystemExit(session.return_code())


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        serve(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "lead":
        lead(sys.argv[2:])
        return
    parser = argparse.ArgumentParser(description="Start launch/strix_vla.launch.py")
    parser.add_argument("--arm", required=True, choices=("true", "false"))
    parser.add_argument("--cameras", required=True, choices=("true", "false"))
    parser.add_argument("--digital-twin", default="false", choices=("true", "false"))
    parser.add_argument("--dry-run", default="false", choices=("true", "false"))
    parser.add_argument("--config", default=None)
    flag = {"true": True, "false": False}
    args = parser.parse_args()
    exec_launch(
        arm=flag[args.arm],
        cameras=flag[args.cameras],
        digital_twin=flag[args.digital_twin],
        dry_run=flag[args.dry_run],
        config=args.config,
    )


if __name__ == "__main__":
    main()
