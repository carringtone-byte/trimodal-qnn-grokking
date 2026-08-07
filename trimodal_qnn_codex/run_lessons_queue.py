from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .config import load_config


STILL_ACTIVE = 259


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform != "win32":
        try:
            import os

            os.kill(pid, 0)
            return True
        except OSError:
            return False
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return False
    exit_code = ctypes.c_ulong()
    try:
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return False
        return int(exit_code.value) == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def read_pid(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def write_event(path: Path, event: str, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": event, "time": time.time(), **payload}, sort_keys=True) + "\n")


def run_logged(command: list[str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n" + "=" * 120 + "\n")
        log.write("started " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
        log.write("command " + json.dumps(command) + "\n")
        log.flush()
        proc = subprocess.Popen(command, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT)
        return proc.wait()


def safe_name(config_path: Path) -> str:
    return config_path.stem


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description="Run trimodal QNN lesson configs, optionally after the mech-interp queue exits.")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "trimodal_qnn_codex/configs/phase1_three_sector_mod97_lessons.yaml",
            "trimodal_qnn_codex/configs/phase1_ordered_route_mod97_lessons.yaml",
        ],
    )
    parser.add_argument("--out-dir", default="trimodal_qnn_codex/outputs/phase1_lessons_queue")
    parser.add_argument("--wait-pid-file", default="tri_modal_modular_grokking/analysis/phase8_seed_40k_mech_interp/queue.pid")
    parser.add_argument("--poll-sec", type=int, default=60)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    out_dir = root / args.out_dir
    status_path = out_dir / "status.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = [root / item for item in args.configs]
    write_event(status_path, "queue_start", configs=[str(path.relative_to(root)) for path in configs])

    wait_pid_path = root / args.wait_pid_file
    waited_pid = read_pid(wait_pid_path)
    if waited_pid and pid_alive(waited_pid):
        write_event(status_path, "wait_start", pid=waited_pid, pid_file=str(wait_pid_path.relative_to(root)))
        while pid_alive(waited_pid):
            time.sleep(max(5, int(args.poll_sec)))
        write_event(status_path, "wait_done", pid=waited_pid)
    else:
        write_event(status_path, "wait_skipped", pid=waited_pid)

    results: list[dict[str, Any]] = []
    start = time.time()
    for config_path in configs:
        cfg = load_config(config_path)
        run_out = root / str(cfg["output_dir"])
        summary_path = run_out / "summary.json"
        name = safe_name(config_path)
        if summary_path.exists() and not args.force:
            write_event(status_path, "skip_done", config=str(config_path.relative_to(root)), output_dir=str(run_out.relative_to(root)))
            results.append({"config": str(config_path.relative_to(root)), "output_dir": str(run_out.relative_to(root)), "skipped": True})
            continue
        command = [sys.executable, "-m", "trimodal_qnn_codex.train", "--config", str(config_path.relative_to(root))]
        write_event(status_path, "start_run", config=str(config_path.relative_to(root)), output_dir=str(run_out.relative_to(root)), command=command)
        run_start = time.time()
        code = run_logged(command, root, out_dir / "logs" / f"{name}.log")
        elapsed = time.time() - run_start
        write_event(status_path, "done_run", config=str(config_path.relative_to(root)), exit_code=code, elapsed_sec=elapsed)
        result: dict[str, Any] = {
            "config": str(config_path.relative_to(root)),
            "output_dir": str(run_out.relative_to(root)),
            "exit_code": code,
            "elapsed_sec": elapsed,
        }
        if summary_path.exists():
            result["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
        results.append(result)
        if code != 0 and not args.continue_on_error:
            break

    queue_summary = {"elapsed_sec": time.time() - start, "results": results}
    (out_dir / "queue_summary.json").write_text(json.dumps(queue_summary, indent=2, sort_keys=True), encoding="utf-8")
    write_event(status_path, "queue_done", elapsed_sec=queue_summary["elapsed_sec"], runs=len(results))
    if any(int(result.get("exit_code", 0)) != 0 for result in results) and not args.continue_on_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
