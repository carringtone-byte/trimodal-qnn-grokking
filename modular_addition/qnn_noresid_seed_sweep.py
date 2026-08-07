from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SweepEntry:
    frequency: int
    seed_index: int
    train_seed: int
    run_dir: Path
    config: Path | None
    source: str


ENTRIES = [
    SweepEntry(
        frequency=13,
        seed_index=0,
        train_seed=9511,
        run_dir=Path("runs/modular_addition_qnn_mod97_expval_delta_k13_noresid_boundary_seed0"),
        config=Path("configs/modular_addition_qnn_mod97_expval_delta_k13_noresid_boundary_seed0.yaml"),
        source="new",
    ),
    SweepEntry(
        frequency=13,
        seed_index=1,
        train_seed=9512,
        run_dir=Path("runs/modular_addition_qnn_mod97_expval_delta_k13_noresid_boundary_seed1"),
        config=Path("configs/modular_addition_qnn_mod97_expval_delta_k13_noresid_boundary_seed1.yaml"),
        source="new",
    ),
    SweepEntry(
        frequency=13,
        seed_index=2,
        train_seed=9513,
        run_dir=Path("runs/modular_addition_qnn_mod97_expval_delta_k13_noresid_boundary_seed2"),
        config=Path("configs/modular_addition_qnn_mod97_expval_delta_k13_noresid_boundary_seed2.yaml"),
        source="new",
    ),
    SweepEntry(
        frequency=21,
        seed_index=0,
        train_seed=9501,
        run_dir=Path("runs/modular_addition_qnn_mod97_expval_delta_k21_noresid_boundary_norefresh"),
        config=Path("configs/modular_addition_qnn_mod97_expval_delta_k21_noresid_boundary.yaml"),
        source="existing_seed0",
    ),
    SweepEntry(
        frequency=21,
        seed_index=1,
        train_seed=9502,
        run_dir=Path("runs/modular_addition_qnn_mod97_expval_delta_k21_noresid_boundary_seed1"),
        config=Path("configs/modular_addition_qnn_mod97_expval_delta_k21_noresid_boundary_seed1.yaml"),
        source="new",
    ),
    SweepEntry(
        frequency=21,
        seed_index=2,
        train_seed=9503,
        run_dir=Path("runs/modular_addition_qnn_mod97_expval_delta_k21_noresid_boundary_seed2"),
        config=Path("configs/modular_addition_qnn_mod97_expval_delta_k21_noresid_boundary_seed2.yaml"),
        source="new",
    ),
]


def load_summary(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def variant_row(summary: dict[str, Any]) -> dict[str, Any] | None:
    for row in summary.get("variants", []):
        if row.get("variant") == "expval_head":
            return row
    return None


def metric_records(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "metrics_expval_head.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def completed(entry: SweepEntry) -> bool:
    summary = load_summary(entry.run_dir)
    row = variant_row(summary) if summary else None
    return bool(row and (entry.run_dir / "checkpoint_expval_head_best.pt").exists())


def command_for(entry: SweepEntry, device: str) -> list[str]:
    if entry.config is None:
        raise ValueError(f"{entry.run_dir} has no runnable config")
    cmd = [
        sys.executable,
        "-m",
        "modular_addition.qnn_mod97",
        "--config",
        str(entry.config),
        "--variants",
        "expval_head",
        "--device",
        device,
        "--disable-progress",
    ]
    if not (entry.run_dir / "summary.json").exists() and (entry.run_dir / "checkpoint_expval_head_best.pt").exists():
        cmd.extend(["--resume-run-dir", str(entry.run_dir), "--resume-kind", "best", "--append-metrics"])
    return cmd


def run_missing(out_dir: Path, device: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for entry in ENTRIES:
        if completed(entry):
            print(f"skip completed k{entry.frequency} seed{entry.seed_index}: {entry.run_dir}", flush=True)
            continue
        if entry.config is None or not entry.config.exists():
            print(f"skip missing config k{entry.frequency} seed{entry.seed_index}: {entry.config}", flush=True)
            continue
        entry.run_dir.mkdir(parents=True, exist_ok=True)
        cmd = command_for(entry, device)
        print("run " + " ".join(cmd), flush=True)
        with (entry.run_dir / "train_stdout.log").open("a", encoding="utf-8") as stdout, (
            entry.run_dir / "train_stderr.log"
        ).open("a", encoding="utf-8") as stderr:
            stdout.write("\n# " + " ".join(cmd) + "\n")
            stderr.write("\n# " + " ".join(cmd) + "\n")
            result = subprocess.run(cmd, stdout=stdout, stderr=stderr, check=False)
        if result.returncode != 0:
            raise SystemExit(f"run failed with code {result.returncode}: {entry.run_dir}")
        aggregate(out_dir)


def row_for(entry: SweepEntry) -> dict[str, Any]:
    summary = load_summary(entry.run_dir)
    row = variant_row(summary) if summary else None
    best = row.get("best_record", {}) if row else {}
    metric_rows = metric_records(entry.run_dir) if not row else []
    partial_best = max(metric_rows, key=lambda item: float(item.get("test_accuracy", 0.0))) if metric_rows else None
    if row:
        status = "completed"
        accuracy = float(row.get("best_test_accuracy", 0.0))
        best_record = best
        completed_step = row.get("completed_step", "")
        stopped_early = row.get("stopped_early", "")
        early_stop_reason = row.get("early_stop_reason", "")
    elif partial_best:
        status = "suspended"
        accuracy = float(partial_best.get("test_accuracy", 0.0))
        best_record = partial_best
        completed_step = metric_rows[-1].get("step", "")
        stopped_early = "interrupted"
        early_stop_reason = "suspended by user before summary.json was written"
    else:
        status = "pending"
        accuracy = 0.0
        best_record = {}
        completed_step = ""
        stopped_early = ""
        early_stop_reason = ""
    examples = int(float(best.get("test_examples", 6587)))
    if partial_best:
        examples = int(float(partial_best.get("test_examples", examples)))
    return {
        "frequency": entry.frequency,
        "seed_index": entry.seed_index,
        "train_seed": entry.train_seed,
        "status": status,
        "source": entry.source,
        "run_dir": str(entry.run_dir),
        "config": str(entry.config) if entry.config else "",
        "best_step": best_record.get("step", ""),
        "completed_step": completed_step,
        "best_test_accuracy": accuracy if status != "pending" else "",
        "best_test_errors": int(round((1.0 - accuracy) * examples)) if status != "pending" else "",
        "train_accuracy_at_best": best_record.get("train_full_accuracy", ""),
        "wrap_accuracy_at_best": best_record.get("test_wrap_accuracy", ""),
        "nowrap_accuracy_at_best": best_record.get("test_nowrap_accuracy", ""),
        "stopped_early": stopped_early,
        "early_stop_reason": early_stop_reason,
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[int, list[float]] = {}
    for row in rows:
        if row["status"] != "completed":
            continue
        groups.setdefault(int(row["frequency"]), []).append(float(row["best_test_accuracy"]))
    out = []
    for frequency, vals in sorted(groups.items()):
        out.append(
            {
                "frequency": frequency,
                "seeds": len(vals),
                "held_out_mean": statistics.fmean(vals),
                "held_out_std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "held_out_min": min(vals),
                "held_out_max": max(vals),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> list[str]:
    lines = [
        "| " + " | ".join(label for label, _ in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(key, "")) for _, key in columns) + " |")
    return lines


def write_report(path: Path, rows: list[dict[str, Any]], aggregates: list[dict[str, Any]]) -> None:
    lines = [
        "# QNN No-Residual Boundary Seed Sweep",
        "",
        "This sweep tests whether the remaining no-residual Fourier-delta QNN errors are stable across continuation seeds.",
        "It is a warm-start continuation sweep from the existing `k<=21` native checkpoint, not a fully independent source-checkpoint seed sweep.",
        "",
        "## Per-Run Status",
        "",
        *markdown_table(
            rows,
            [
                ("k", "frequency"),
                ("seed", "seed_index"),
                ("status", "status"),
                ("best step", "best_step"),
                ("held-out", "best_test_accuracy"),
                ("errors", "best_test_errors"),
                ("train", "train_accuracy_at_best"),
                ("wrap", "wrap_accuracy_at_best"),
                ("no-wrap", "nowrap_accuracy_at_best"),
            ],
        ),
        "",
        "## Aggregate",
        "",
        *markdown_table(
            aggregates,
            [
                ("k", "frequency"),
                ("seeds", "seeds"),
                ("held-out mean", "held_out_mean"),
                ("held-out std", "held_out_std"),
                ("held-out min", "held_out_min"),
                ("held-out max", "held_out_max"),
            ],
        ),
        "",
        "## Interpretation Guardrail",
        "",
        "If the same error families recur across seeds, the residual failures are probably structural for this warm-start recipe.",
        "If the failures vary strongly by seed, then the remaining aliases are more likely optimization accidents.",
        "A later source-checkpoint seed sweep is still needed for a stronger claim.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def aggregate(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [row_for(entry) for entry in ENTRIES]
    aggregates = aggregate_rows(rows)
    write_csv(out_dir / "sweep_status.csv", rows)
    write_csv(out_dir / "sweep_aggregate.csv", aggregates)
    (out_dir / "sweep_summary.json").write_text(
        json.dumps({"runs": rows, "aggregate": aggregates}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(out_dir / "NORESID_SEED_SWEEP_SUMMARY.md", rows, aggregates)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and aggregate the QNN no-residual boundary seed sweep.")
    parser.add_argument("--out-dir", default="analysis/qnn_noresid_seed_sweep")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-missing", action="store_true")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    if args.run_missing:
        run_missing(out_dir, args.device)
    aggregate(out_dir)
    print(out_dir / "NORESID_SEED_SWEEP_SUMMARY.md")


if __name__ == "__main__":
    main()
