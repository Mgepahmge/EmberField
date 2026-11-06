"""Linux PTY integration: real terminal lifecycle, cancellation and numerical equivalence."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import pty
import select
import signal
import struct
import subprocess
import tempfile
import termios
import time


def terminal_run(command, columns=112, rows=40, actions=(), timeout=30):
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", rows, columns, 0, 0))
    before = termios.tcgetattr(slave)
    environment = dict(os.environ, TERM="xterm-256color")
    process = subprocess.Popen(
        [str(x) for x in command], stdin=slave, stdout=slave, stderr=slave,
        env=environment, start_new_session=True,
        preexec_fn=lambda: fcntl.ioctl(0, termios.TIOCSCTTY, 0),
    )
    output = bytearray()
    pending = list(actions)
    started = time.monotonic()
    try:
        while process.poll() is None:
            elapsed = time.monotonic() - started
            assert elapsed < timeout, (command, output[-3000:])
            while pending and elapsed >= pending[0][0]:
                _, action = pending.pop(0)
                if isinstance(action, bytes):
                    os.write(master, action)
                elif isinstance(action, tuple):
                    width, height = action
                    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", height, width, 0, 0))
                    process.send_signal(signal.SIGWINCH)
                else:
                    process.send_signal(action)
            if select.select([master], [], [], 0.025)[0]:
                output.extend(os.read(master, 65536))
        while select.select([master], [], [], 0)[0]:
            output.extend(os.read(master, 65536))
        after = termios.tcgetattr(slave)
        # FTXUI must restore canonical input, echo, signal handling and output flags.
        assert before == after, (before, after)
        return process.returncode, bytes(output), time.monotonic() - started
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        os.close(master)
        os.close(slave)


def signature(directory):
    summary = json.loads((directory / "run-summary.json").read_text())
    assert summary["status"] == "completed", summary["status"]
    frames = [(item["step"], (directory / item["path"]).read_bytes()) for item in summary["frames"]]
    with (directory / "steps.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    rows = [{key: value for key, value in row.items() if not key.endswith("_s")} for row in rows]
    return summary["lastAcceptedStep"], summary["lastPersistedStep"], frames, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    parser.add_argument("cases", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--artifacts", type=Path)
    args = parser.parse_args()
    executable = args.executable.resolve()
    cases = args.cases.resolve()
    for flag in ["--help", "--version"]:
        result = subprocess.run([executable, flag], capture_output=True)
        assert result.returncode == 0 and b"\x1b" not in result.stdout
    with tempfile.TemporaryDirectory(prefix="emberfield-tui-") as temporary:
        root = Path(temporary)
        def command(case, output, *flags):
            return [executable, case, "--output-root", root, "--output", output, *flags]

        for name in ["steady", "transient", "time-varying"]:
            case = cases / (name + ".json")
            plain = subprocess.run(command(case, name + "-plain"), capture_output=True)
            assert plain.returncode == 0 and b"\x1b" not in plain.stdout, plain
            code, transcript, _ = terminal_run(command(case, name + "-tui", "--ui", "tui", "--no-wait"))
            assert code == 0 and b"\x1b[?1049h" in transcript and b"\x1b[?1049l" in transcript, transcript[-3000:]
            assert signature(root / (name + "-plain")) == signature(root / (name + "-tui"))
            if args.baseline:
                result = subprocess.run([args.baseline, case, "--output-root", root, "--output", name + "-base"], capture_output=True)
                assert result.returncode == 0, result
                assert signature(root / (name + "-base")) == signature(root / (name + "-plain"))

        # Default completed screen remains until the user closes it.
        code, transcript, elapsed = terminal_run(command(cases / "steady.json", "wait", "--ui", "tui"), actions=[(1.5, b"q")])
        assert code == 0 and elapsed >= 1.5
        if args.artifacts:
            args.artifacts.mkdir(parents=True, exist_ok=True)
            (args.artifacts / "wide.ansi").write_bytes(transcript)

        # Minimum supported size and a live resize use the same run, never restart it.
        code, narrow, _ = terminal_run(command(cases / "transient.json", "narrow", "--ui", "tui"), 40, 18, [(1.2, b"q")])
        assert code == 0
        if args.artifacts:
            (args.artifacts / "narrow.ansi").write_bytes(narrow)
        code, resized, _ = terminal_run(command(cases / "steady.json", "resize", "--ui", "tui"), actions=[(0.6, (30, 10))])
        assert code == 0 and "终端过小".encode() in resized

        # Long enough to cancel during computation, bounded cursor preflight.
        config = json.loads((cases / "transient.json").read_text())
        config["mesh"]["path"] = str(cases / "plate.msh")
        config["time"] = {"start": 0, "end": 1000, "step": 0.01, "maxSteps": 100001}
        config["output"]["includeInitial"] = False
        long_case = root / "long.json"
        long_case.write_text(json.dumps(config))
        for label, actions in [
            ("q", [(0.5, b"q")]), ("escape", [(0.5, b"\x1b")]), ("ctrl-c", [(0.5, b"\x03")]),
            ("sigint", [(0.5, signal.SIGINT), (0.6, signal.SIGINT)]),
            ("sigterm", [(0.5, signal.SIGTERM), (0.6, signal.SIGTERM)]),
        ]:
            code, transcript, _ = terminal_run(command(long_case, label, "--ui", "tui", "--no-wait"), actions=actions)
            assert code == 130, (label, code, transcript[-3000:])
            summary = root / label / "run-summary.json"
            if summary.exists():
                document = json.loads(summary.read_text())
                assert document["status"] != "completed"
        # Closing the PTY sends SIGHUP; the worker still exits through safe cancellation.
        master, slave = pty.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 112, 0, 0))
        process = subprocess.Popen(command(long_case, "disconnect", "--ui", "tui"),
            stdin=slave, stdout=slave, stderr=slave, env=dict(os.environ, TERM="xterm"),
            start_new_session=True, preexec_fn=lambda: fcntl.ioctl(0, termios.TIOCSCTTY, 0))
        time.sleep(0.6)
        os.close(master)
        try:
            assert process.wait(timeout=20) == 130
        finally:
            if process.poll() is None:
                process.kill(); process.wait()
            os.close(slave)
        # Explicit TUI on redirected streams must reject before creating output.
        result = subprocess.run(command(cases / "steady.json", "reject", "--ui", "tui"), capture_output=True)
        assert result.returncode != 0 and not (root / "reject").exists()
        for index, environment in enumerate([dict(os.environ, TERM="dumb"), dict(os.environ, TERM="xterm")]):
            result = subprocess.run(command(cases / "steady.json", "redirect-" + str(index)), env=environment, capture_output=True)
            assert result.returncode == 0 and b"\x1b" not in result.stdout
    print("PTY lifecycle, cancellation, resize, redirection and numerical equivalence passed")


if __name__ == "__main__":
    main()
