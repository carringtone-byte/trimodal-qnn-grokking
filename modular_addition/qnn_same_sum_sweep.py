from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Entry:
    architecture: str
    kind: str
    run_dir: Path
    config: Path | None
    control_run: Path | None = None


ENTRIES = [
    Entry(
        architecture="aux_control",
        kind="control",
        run_dir=Path("runs/modular_addition_qnn_pub_mod97_aux_seed0"),
        config=None,
    ),
    Entry(
        architecture="aux_control",
        kind="same_sum",
        run_dir=Path("runs/modular_addition_qnn_mod97_same_sum_aux_control_seed0"),
        config=Path("configs/modular_addition_qnn_mod97_same_sum_aux_control_seed0.yaml"),
        control_run=Path("runs/modular_addition_qnn_pub_mod97_aux_seed0"),
    ),
    Entry(
        architecture="mean",
        kind="control",
        run_dir=Path("runs/modular_addition_qnn_pub_mod97_mean_seed0"),
        config=None,
    ),
    Entry(
        architecture="mean",
        kind="same_sum",
        run_dir=Path("runs/modular_addition_qnn_mod97_same_sum_mean_seed0"),
        config=Path("configs/modular_addition_qnn_mod97_same_sum_mean_seed0.yaml"),
        control_run=Path("runs/modular_addition_qnn_pub_mod97_mean_seed0"),
    ),
    Entry(
        architecture="residual",
        kind="control",
        run_dir=Path("runs/modular_addition_qnn_pub_mod97_residual_seed0"),
        config=None,
    ),
    Entry(
        architecture="residual",
        kind="same_sum",
        run_dir=Path("runs/modular_addition_qnn_mod97_same_sum_residual_seed0"),
        config=Path("configs/modular_addition_qnn_mod97_same_sum_residual_seed0.yaml"),
        control_run=Path("runs/modular_addition_qnn_pub_mod97_residual_seed0"),
    ),
]


def load_summary(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def variant_row(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if summary is None:
        return None
    for row in summary.get("variants", []):
        if row.get("variant") == "prob_head":
            return row
    return None


def metric_records(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "metrics_prob_head.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def completed(entry: Entry) -> bool:
    row = variant_row(load_summary(entry.run_dir))
    return bool(row and (entry.run_dir / "checkpoint_prob_head_best.pt").exists())


def command_for(entry: Entry, device: str) -> list[str]:
    if entry.config is None:
        raise ValueError(f"entry has no runnable config: {entry}")
    cmd = [
        sys.executable,
        "-m",
        "modular_addition.qnn_mod97",
        "--config",
        str(entry.config),
        "--variants",
        "prob_head",
        "--device",
        device,
        "--disable-progress",
    ]
    if not (entry.run_dir / "summary.json").exists() and (entry.run_dir / "checkpoint_prob_head_best.pt").exists():
        cmd.extend(["--resume-run-dir", str(entry.run_dir), "--resume-kind", "best", "--append-metrics"])
    return cmd


def run_missing(out_dir: Path, device: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for entry in ENTRIES:
        if entry.kind == "control":
            continue
        if completed(entry):
            print(f"skip completed {entry.architecture}: {entry.run_dir}", flush=True)
            continue
        if entry.config is None or not entry.config.exists():
            print(f"skip missing config {entry.architecture}: {entry.config}", flush=True)
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


def best_available(entry: Entry) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None]:
    row = variant_row(load_summary(entry.run_dir))
    if row:
        return "completed", row, row.get("best_record", {})
    records = metric_records(entry.run_dir)
    if records:
        best = max(records, key=lambda item: float(item.get("test_accuracy", 0.0)))
        partial = {
            "best_test_accuracy": best.get("test_accuracy", 0.0),
            "final_train_accuracy": best.get("train_full_accuracy", 0.0),
            "completed_step": records[-1].get("step", ""),
            "stopped_early": "interrupted",
        }
        return "suspended", partial, best
    return "pending", None, None


def result_row(entry: Entry) -> dict[str, Any]:
    status, row, best = best_available(entry)
    accuracy = float(row.get("best_test_accuracy", 0.0)) if row else None
    examples = int(float(best.get("test_examples", 6587))) if best else 6587
    return {
        "architecture": entry.architecture,
        "kind": entry.kind,
        "status": status,
        "run_dir": str(entry.run_dir),
        "control_run": str(entry.control_run) if entry.control_run else "",
        "best_step": best.get("step", "") if best else "",
        "completed_step": row.get("completed_step", "") if row else "",
        "held_out_accuracy": accuracy if accuracy is not None else "",
        "held_out_errors": int(round((1.0 - accuracy) * examples)) if accuracy is not None else "",
        "train_accuracy_at_best": best.get("train_full_accuracy", "") if best else "",
        "wrap_accuracy_at_best": best.get("test_wrap_accuracy", "") if best else "",
        "nowrap_accuracy_at_best": best.get("test_nowrap_accuracy", "") if best else "",
        "consistency_loss_at_best": best.get("train_consistency_loss", "") if best else "",
    }


def comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_arch: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_arch.setdefault(str(row["architecture"]), {})[str(row["kind"])] = row
    out = []
    for arch, group in sorted(by_arch.items()):
        control = group.get("control")
        same = group.get("same_sum")
        if not control or not same:
            continue
        if control["held_out_accuracy"] == "" or same["held_out_accuracy"] == "":
            gain = ""
        else:
            gain = float(same["held_out_accuracy"]) - float(control["held_out_accuracy"])
        out.append(
            {
                "architecture": arch,
                "control_accuracy": control["held_out_accuracy"],
                "same_sum_accuracy": same["held_out_accuracy"],
                "gain": gain,
                "control_errors": control["held_out_errors"],
                "same_sum_errors": same["held_out_errors"],
                "same_sum_status": same["status"],
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


def table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> list[str]:
    lines = [
        "| " + " | ".join(label for label, _ in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(key, "")) for _, key in columns) + " |")
    return lines


def write_report(path: Path, rows: list[dict[str, Any]], comparisons: list[dict[str, Any]]) -> None:
    lines = [
        "# QNN Same-Sum Invariance Seed-0 Sweep",
        "",
        "This first-pass test adds a global same-sum KL/CE objective to seed-0, split-30, mod-97 layerwise QNNs.",
        "The controls are the existing seed-0 publication-sweep runs with the same train/test split.",
        "",
        "## Comparisons",
        "",
        *table(
            comparisons,
            [
                ("architecture", "architecture"),
                ("control", "control_accuracy"),
                ("same-sum", "same_sum_accuracy"),
                ("gain", "gain"),
                ("control errors", "control_errors"),
                ("same-sum errors", "same_sum_errors"),
                ("status", "same_sum_status"),
            ],
        ),
        "",
        "## Run Status",
        "",
        *table(
            rows,
            [
                ("architecture", "architecture"),
                ("kind", "kind"),
                ("status", "status"),
                ("best step", "best_step"),
                ("held-out", "held_out_accuracy"),
                ("train", "train_accuracy_at_best"),
                ("wrap", "wrap_accuracy_at_best"),
                ("no-wrap", "nowrap_accuracy_at_best"),
                ("consistency", "consistency_loss_at_best"),
            ],
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def aggregate(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [result_row(entry) for entry in ENTRIES]
    comparisons = comparison_rows(rows)
    write_csv(out_dir / "same_sum_status.csv", rows)
    write_csv(out_dir / "same_sum_comparison.csv", comparisons)
    (out_dir / "same_sum_summary.json").write_text(
        json.dumps({"runs": rows, "comparisons": comparisons}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(out_dir / "QNN_SAME_SUM_SWEEP_SUMMARY.md", rows, comparisons)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and aggregate seed-0 QNN same-sum invariance tests.")
    parser.add_argument("--out-dir", default="analysis/qnn_same_sum_sweep")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-missing", action="store_true")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    if args.run_missing:
        run_missing(out_dir, args.device)
    aggregate(out_dir)
    print(out_dir / "QNN_SAME_SUM_SWEEP_SUMMARY.md")


if __name__ == "__main__":
    main()

