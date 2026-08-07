from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_SELECTION = "aux,adapter,residual,fourier_delta_matched,fourier_product_delta"


def parse_list(text: str) -> set[str]:
    return {part.strip() for part in text.split(",") if part.strip()}


def parse_ints(text: str) -> set[int]:
    return {int(part.strip()) for part in text.split(",") if part.strip()}


def parse_floats(text: str) -> set[float]:
    return {float(part.strip()) for part in text.split(",") if part.strip()}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def refresh_status(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "-m",
        "modular_addition.qnn_publication_sweeps",
        "--write-configs",
        "--moduli",
        args.moduli,
        "--seeds",
        args.seeds,
        "--split-seeds",
        args.split_seeds,
        "--train-fractions",
        args.train_fractions,
        "--architectures",
        args.qnn_architectures,
        "--baselines",
        args.baselines,
        "--out-dir",
        str(args.out_dir),
        "--device",
        args.device,
    ]
    if args.no_baselines:
        cmd.append("--no-baselines")
    subprocess.run(cmd, check=True)


def selected_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows = read_csv(args.out_dir / "sweep_status.csv")
    selected = parse_list(args.selection)
    moduli = parse_ints(args.moduli)
    seeds = parse_ints(args.seeds)
    split_seeds = parse_ints(args.split_seeds)
    train_fractions = parse_floats(args.train_fractions)
    out: list[dict[str, str]] = []
    for row in rows:
        try:
            modulus = int(row.get("modulus", ""))
            seed = int(row.get("seed", ""))
            split_seed = int(row.get("split_seed", "0") or 0)
            train_fraction = float(row.get("train_fraction", "0.3") or 0.3)
        except ValueError:
            continue
        if modulus not in moduli or seed not in seeds:
            continue
        if split_seed not in split_seeds:
            continue
        if not any(abs(train_fraction - wanted) < 1.0e-9 for wanted in train_fractions):
            continue
        if row.get("architecture") not in selected:
            continue
        if row.get("status") == "complete" and not args.rerun_complete:
            continue
        out.append(row)
    return out


def log_name(row: dict[str, str]) -> str:
    train_fraction = float(row.get("train_fraction", "0.3") or 0.3)
    frac = f"train{int(round(train_fraction * 100)):02d}"
    split = row.get("split_seed", "0") or "0"
    return f"{row['kind']}_mod{row['modulus']}_{row['architecture']}_{frac}_split{split}_seed{row['seed']}.log"


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def command_for_row(row: dict[str, str]) -> tuple[str, str | None]:
    command = row["command"]
    run_dir = Path(row.get("run_dir", ""))
    if row.get("kind") == "qnn" and run_dir.exists() and not (run_dir / "summary.json").exists():
        best_checkpoint = run_dir / "checkpoint_prob_head_best.pt"
        if best_checkpoint.exists() and "--resume-run-dir" not in command:
            command = f"{command} --resume-run-dir {run_dir} --resume-kind best --append-metrics"
            return command, str(best_checkpoint)
    return command, None


def run_one(row: dict[str, str], args: argparse.Namespace) -> int:
    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / log_name(row)
    record_path = args.out_dir / "subset_runner_status.jsonl"
    started = time.time()
    command, resumed_from = command_for_row(row)
    append_jsonl(
        record_path,
        {
            "event": "start",
            "time": started,
            "kind": row.get("kind"),
            "modulus": row.get("modulus"),
            "train_fraction": row.get("train_fraction"),
            "split_seed": row.get("split_seed"),
            "seed": row.get("seed"),
            "architecture": row.get("architecture"),
            "command": command,
            "resumed_from": resumed_from,
            "log": str(log_path),
        },
    )
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {command}\n\n")
        log.flush()
        result = subprocess.run(command, shell=True, stdout=log, stderr=subprocess.STDOUT, text=True)
    ended = time.time()
    append_jsonl(
        record_path,
        {
            "event": "end",
            "time": ended,
            "elapsed_sec": ended - started,
            "returncode": result.returncode,
            "kind": row.get("kind"),
            "modulus": row.get("modulus"),
            "train_fraction": row.get("train_fraction"),
            "split_seed": row.get("split_seed"),
            "seed": row.get("seed"),
            "architecture": row.get("architecture"),
            "log": str(log_path),
        },
    )
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a selected subset of the QNN publication sweep manifest.")
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/qnn_publication_sweep"))
    parser.add_argument("--log-dir", type=Path, default=Path("analysis/qnn_publication_sweep/logs"))
    parser.add_argument("--selection", default=DEFAULT_SELECTION)
    parser.add_argument("--moduli", default="31,97,127")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--split-seeds", default="0")
    parser.add_argument("--train-fractions", default="0.3")
    parser.add_argument("--qnn-architectures", default="aux,adapter,residual")
    parser.add_argument(
        "--baselines",
        default="fourier_linear,fourier_mlp_matched,fourier_delta_matched,fourier_product_linear,fourier_product_delta,raw_numeric_mlp_matched",
    )
    parser.add_argument("--no-baselines", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--rerun-complete", action="store_true")
    parser.add_argument("--stop-on-fail", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    refresh_status(args)
    rows = selected_rows(args)
    if args.max_runs is not None:
        rows = rows[: args.max_runs]
    print(f"Selected pending runs: {len(rows)}", flush=True)
    for row in rows:
        print(
            f"Running {row['kind']} mod{row['modulus']} {row['architecture']} seed{row['seed']}",
            flush=True,
        )
        if args.dry_run:
            print(row["command"], flush=True)
            continue
        returncode = run_one(row, args)
        refresh_status(args)
        if returncode != 0:
            print(f"Run failed with return code {returncode}: {row['command']}", flush=True)
            if args.stop_on_fail:
                raise SystemExit(returncode)
    refresh_status(args)


if __name__ == "__main__":
    main()
