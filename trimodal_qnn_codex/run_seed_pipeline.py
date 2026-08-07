from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from .config import load_config, write_config


DEFAULT_SEEDS = (9302, 9303, 9304, 9305)
CHECKPOINT_RE = re.compile(r"checkpoint_(\d+)\.pt$")


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def root_path(root: Path, path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else root / path


def relpath(root: Path, path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def posix_join(*parts: str | Path) -> str:
    text_parts = [str(part).strip("/\\") for part in parts if str(part)]
    if not text_parts:
        return ""
    prefix = ""
    if Path(str(parts[0])).is_absolute():
        prefix = str(parts[0]).rstrip("/\\")
        text_parts = [str(part).strip("/\\") for part in parts[1:] if str(part)]
        return "/".join([prefix, *text_parts]).replace("\\", "/")
    return "/".join(text_parts).replace("\\", "/")


def seed_config_name(base_config: Path, seed: int) -> str:
    return f"{base_config.stem}_seed{int(seed)}.yaml"


def seed_run_name(seed: int) -> str:
    return f"seed_{int(seed)}"


def make_seed_config(
    base_cfg: dict[str, Any],
    *,
    seed: int,
    run_root: str,
    device: str | None = None,
    training_steps: int | None = None,
    resume_checkpoint: str | None = None,
) -> dict[str, Any]:
    cfg = deepcopy(base_cfg)
    cfg["seed"] = int(seed)
    cfg["output_dir"] = posix_join(run_root, seed_run_name(seed))
    cfg.setdefault("training", {})
    if device is not None:
        cfg["training"]["device"] = str(device)
    if training_steps is not None:
        cfg["training"]["steps"] = int(training_steps)
    cfg["training"]["resume_checkpoint"] = resume_checkpoint
    cfg.setdefault("notes", {})
    cfg["notes"]["seed_sweep"] = "phase1 strict three-sector layerwise Dirac-mean robustness"
    cfg["notes"]["seed_sweep_base_seed"] = int(base_cfg.get("seed", -1))
    cfg["notes"]["seed_sweep_seed"] = int(seed)
    cfg["notes"]["seed_sweep_resume_checkpoint"] = resume_checkpoint
    return cfg


def write_event(path: Path, event: str, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"event": event, "time": time.time(), **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def run_logged(command: list[str], *, cwd: Path, log_path: Path, dry_run: bool = False) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n" + "=" * 120 + "\n")
        log.write("started " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
        log.write("command " + json.dumps(command) + "\n")
        if dry_run:
            log.write("dry_run true\n")
            return 0
        log.flush()
        proc = subprocess.Popen(command, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT)
        return int(proc.wait())


def discover_regular_checkpoints(run_dir: Path) -> dict[int, Path]:
    checkpoints: dict[int, Path] = {}
    if not run_dir.exists():
        return checkpoints
    for path in run_dir.glob("checkpoint_*.pt"):
        match = CHECKPOINT_RE.match(path.name)
        if match:
            checkpoints[int(match.group(1))] = path
    return dict(sorted(checkpoints.items()))


def latest_regular_checkpoint(run_dir: Path) -> tuple[int, Path] | None:
    checkpoints = discover_regular_checkpoints(run_dir)
    if not checkpoints:
        return None
    step = max(checkpoints)
    return step, checkpoints[step]


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def read_metric_rows(run_dir: Path) -> list[dict[str, Any]]:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"event": "json_decode_error", "raw": line})
    return rows


def choose_best_checkpoint(run_dir: Path, preferred_step: int | None = None) -> dict[str, Any] | None:
    checkpoints = discover_regular_checkpoints(run_dir)
    if not checkpoints:
        return None
    rows = read_metric_rows(run_dir)
    by_step: dict[int, dict[str, Any]] = {}
    for row in rows:
        step_value = row.get("step")
        if step_value is None:
            continue
        try:
            step = int(step_value)
        except (TypeError, ValueError):
            continue
        if step in checkpoints:
            by_step[step] = row

    if preferred_step is not None:
        step = int(preferred_step)
        if step not in checkpoints:
            return None
        row = by_step.get(step, {})
        return checkpoint_summary(step, checkpoints[step], row, "preferred")

    candidates: list[tuple[float, int, dict[str, Any]]] = []
    for step, row in by_step.items():
        heldout_accuracy = safe_float(row.get("heldout_accuracy"))
        if math.isfinite(heldout_accuracy):
            candidates.append((heldout_accuracy, step, row))
    if candidates:
        _, step, row = max(candidates, key=lambda item: (item[0], item[1]))
        return checkpoint_summary(step, checkpoints[step], row, "best_heldout_accuracy")

    step = max(checkpoints)
    return checkpoint_summary(step, checkpoints[step], by_step.get(step, {}), "latest_regular_checkpoint")


def checkpoint_summary(step: int, path: Path, row: dict[str, Any], source: str) -> dict[str, Any]:
    return {
        "step": int(step),
        "path": str(path),
        "source": source,
        "heldout_accuracy": safe_float(row.get("heldout_accuracy")),
        "heldout_loss": safe_float(row.get("heldout_loss")),
        "heldout_cross_ablation_accuracy": safe_float(row.get("heldout_cross_ablation_accuracy")),
        "train_accuracy": safe_float(row.get("train_accuracy")),
        "train_loss": safe_float(row.get("train_loss")),
    }


def command_done(out_dir: Path, required_file: str = "manifest.json") -> bool:
    return (out_dir / required_file).exists()


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_checkpoint_analysis_command(args: argparse.Namespace, root: Path, run_dir: Path, out_dir: Path, step: int) -> list[str]:
    command = [
        args.python_exe,
        "-m",
        "trimodal_qnn_codex.analyze_checkpoints",
        "--run-dir",
        relpath(root, run_dir),
        "--checkpoint-step",
        str(int(step)),
        "--out-dir",
        relpath(root, out_dir),
        "--batch-size",
        str(int(args.checkpoint_batch_size)),
        "--device",
        str(args.device),
        "--cutoffs",
        str(args.cutoffs),
    ]
    if args.include_nonfinite:
        command.append("--include-nonfinite")
    return command


def build_causal_command(args: argparse.Namespace, root: Path, run_dir: Path, out_dir: Path, step: int) -> list[str]:
    return [
        args.python_exe,
        "-m",
        "trimodal_qnn_codex.analyze_dirac_causal",
        "--run-dir",
        relpath(root, run_dir),
        "--checkpoint-step",
        str(int(step)),
        "--out-dir",
        relpath(root, out_dir),
        "--split",
        str(args.causal_split),
        "--batch-size",
        str(int(args.causal_batch_size)),
        "--device",
        str(args.device),
        "--deltas",
        str(args.deltas),
    ]


def build_cca_command(args: argparse.Namespace, root: Path, run_dir: Path, out_dir: Path, step: int) -> list[str]:
    return [
        args.python_exe,
        "-m",
        "trimodal_qnn_codex.analyze_sector_cca",
        "--run-dir",
        relpath(root, run_dir),
        "--checkpoint-step",
        str(int(step)),
        "--out-dir",
        relpath(root, out_dir),
        "--batch-size",
        str(int(args.cca_batch_size)),
        "--device",
        str(args.device),
        "--n-components",
        str(int(args.cca_components)),
        "--cca-reg",
        str(float(args.cca_reg)),
        "--probe-reg",
        str(float(args.probe_reg)),
    ]


def parse_checkpoint_step(value: str) -> int | None:
    if str(value).lower() == "auto":
        return None
    return int(value)


def train_should_run(run_dir: Path, *, force: bool, resume_existing: bool) -> tuple[bool, str]:
    if force:
        return True, "force_train"
    if (run_dir / "summary.json").exists():
        return False, "summary_exists"
    latest = latest_regular_checkpoint(run_dir)
    if latest is not None and not resume_existing:
        return False, "existing_checkpoint"
    return True, "needs_training"


def run_one_seed(args: argparse.Namespace, root: Path, base_cfg: dict[str, Any], base_config_path: Path, seed: int) -> dict[str, Any]:
    config_dir = root_path(root, args.config_dir)
    run_root = root_path(root, args.out_dir)
    analysis_root = root_path(root, args.analysis_root)
    run_dir = run_root / seed_run_name(seed)
    log_dir = run_root / "logs"
    config_path = config_dir / seed_config_name(base_config_path, seed)
    status_path = run_root / "seed_pipeline_status.jsonl"

    resume_checkpoint: str | None = None
    latest = latest_regular_checkpoint(run_dir)
    if args.resume_existing and latest is not None and not args.force_train:
        latest_step, latest_path = latest
        target_steps = int(args.training_steps or base_cfg["training"].get("steps", 0))
        if latest_step < target_steps:
            resume_checkpoint = relpath(root, latest_path)

    cfg = make_seed_config(
        base_cfg,
        seed=seed,
        run_root=relpath(root, run_root),
        device=args.device,
        training_steps=args.training_steps,
        resume_checkpoint=resume_checkpoint,
    )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if args.force_config or not config_path.exists() or args.dry_run:
        write_config(cfg, config_path)
    else:
        write_config(cfg, config_path)

    result: dict[str, Any] = {
        "seed": int(seed),
        "config": relpath(root, config_path),
        "run_dir": relpath(root, run_dir),
        "analysis_root": relpath(root, analysis_root),
        "train_exit_code": None,
        "analysis_exit_codes": {},
        "best_checkpoint": None,
    }
    write_event(status_path, "seed_start", seed=seed, config=result["config"], run_dir=result["run_dir"])

    should_train, train_reason = train_should_run(run_dir, force=args.force_train, resume_existing=args.resume_existing)
    result["train_decision"] = train_reason
    if args.no_train:
        should_train = False
        result["train_decision"] = "disabled"
    if should_train:
        train_command = [args.python_exe, "-m", "trimodal_qnn_codex.train", "--config", relpath(root, config_path)]
        write_event(status_path, "train_start", seed=seed, command=train_command, resume_checkpoint=resume_checkpoint)
        start = time.time()
        code = run_logged(train_command, cwd=root, log_path=log_dir / f"train_seed_{seed}.log", dry_run=args.dry_run)
        elapsed = time.time() - start
        result["train_exit_code"] = int(code)
        result["train_elapsed_sec"] = elapsed
        write_event(status_path, "train_done", seed=seed, exit_code=code, elapsed_sec=elapsed)
        if code != 0 and args.stop_on_error:
            result["stopped_after_train_error"] = True
            write_event(status_path, "seed_stop_train_error", seed=seed, exit_code=code)
            return result
    else:
        write_event(status_path, "train_skip", seed=seed, reason=result["train_decision"])

    if args.no_analyze:
        write_event(status_path, "analysis_skip", seed=seed, reason="disabled")
        write_seed_summary(root, run_dir, analysis_root, result)
        return result

    preferred = parse_checkpoint_step(args.checkpoint_step)
    best = None if args.dry_run else choose_best_checkpoint(run_dir, preferred)
    if args.dry_run:
        best = {"step": preferred or -1, "source": "dry_run", "path": "", "heldout_accuracy": float("nan")}
    if best is None:
        result["analysis_skipped_reason"] = "no_regular_checkpoint"
        write_event(status_path, "analysis_skip", seed=seed, reason="no_regular_checkpoint")
        write_seed_summary(root, run_dir, analysis_root, result)
        return result
    step = int(best["step"])
    result["best_checkpoint"] = best
    write_event(status_path, "checkpoint_selected", seed=seed, checkpoint=best)

    analysis_jobs: list[tuple[str, Path, list[str]]] = []
    checkpoint_out = analysis_root / f"seed_{seed}_step_{step}"
    causal_out = analysis_root / f"seed_{seed}_step_{step}_causal"
    cca_out = analysis_root / f"seed_{seed}_step_{step}_sector_cca"
    if not args.skip_checkpoint_analysis:
        analysis_jobs.append(("checkpoint", checkpoint_out, build_checkpoint_analysis_command(args, root, run_dir, checkpoint_out, step)))
    if not args.skip_causal:
        analysis_jobs.append(("causal", causal_out, build_causal_command(args, root, run_dir, causal_out, step)))
    if not args.skip_cca:
        analysis_jobs.append(("sector_cca", cca_out, build_cca_command(args, root, run_dir, cca_out, step)))

    for name, out_dir, command in analysis_jobs:
        if command_done(out_dir) and not args.force_analysis and not args.dry_run:
            result["analysis_exit_codes"][name] = 0
            result[f"{name}_decision"] = "manifest_exists"
            write_event(status_path, "analysis_skip", seed=seed, analysis=name, reason="manifest_exists", out_dir=relpath(root, out_dir))
            continue
        write_event(status_path, "analysis_start", seed=seed, analysis=name, out_dir=relpath(root, out_dir), command=command)
        start = time.time()
        code = run_logged(command, cwd=root, log_path=log_dir / f"{name}_seed_{seed}_step_{step}.log", dry_run=args.dry_run)
        elapsed = time.time() - start
        result["analysis_exit_codes"][name] = int(code)
        result[f"{name}_elapsed_sec"] = elapsed
        write_event(status_path, "analysis_done", seed=seed, analysis=name, exit_code=code, elapsed_sec=elapsed)
        if code != 0 and args.stop_on_error:
            result["stopped_after_analysis_error"] = name
            break

    write_seed_summary(root, run_dir, analysis_root, result)
    write_event(status_path, "seed_done", seed=seed, result=result)
    return result


def write_seed_summary(root: Path, run_dir: Path, analysis_root: Path, result: dict[str, Any]) -> None:
    summary_path = analysis_root / f"seed_{int(result['seed'])}_pipeline_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload = deepcopy(result)
    payload["run_dir_abs"] = str(run_dir.resolve())
    payload["analysis_root_abs"] = str(analysis_root.resolve())
    payload["metrics_path"] = relpath(root, run_dir / "metrics.jsonl")
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict trimodal QNN seed training and mechanistic interpretation.")
    parser.add_argument("--base-config", default="trimodal_qnn_codex/configs/phase1_three_sector_mod97_dirac_mean.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--config-dir", default="trimodal_qnn_codex/configs/seed_sweeps")
    parser.add_argument("--out-dir", default="trimodal_qnn_codex/outputs/phase1_three_sector_dirac_mean_seed_sweep")
    parser.add_argument("--analysis-root", default="trimodal_qnn_codex/analysis/phase1_three_sector_dirac_mean_seed_sweep")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--training-steps", type=int, default=None)
    parser.add_argument("--checkpoint-step", default="auto", help="Use a fixed checkpoint step, or auto-select best held-out regular checkpoint.")
    parser.add_argument("--resume-existing", action="store_true", help="Resume incomplete seed runs from the latest regular checkpoint.")
    parser.add_argument("--force-config", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-analysis", action="store_true")
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--no-analyze", action="store_true")
    parser.add_argument("--skip-checkpoint-analysis", action="store_true")
    parser.add_argument("--skip-causal", action="store_true")
    parser.add_argument("--skip-cca", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--checkpoint-batch-size", type=int, default=1024)
    parser.add_argument("--causal-batch-size", type=int, default=512)
    parser.add_argument("--cca-batch-size", type=int, default=1024)
    parser.add_argument("--causal-split", choices=["heldout", "train", "all"], default="heldout")
    parser.add_argument("--cutoffs", default="1,2,3,5,8,13,21")
    parser.add_argument("--deltas", default="1,2,5,13")
    parser.add_argument("--include-nonfinite", action="store_true")
    parser.add_argument("--cca-components", type=int, default=32)
    parser.add_argument("--cca-reg", type=float, default=1e-3)
    parser.add_argument("--probe-reg", type=float, default=1e-2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    base_config_path = root_path(root, args.base_config)
    base_cfg = load_config(base_config_path)
    run_root = root_path(root, args.out_dir)
    analysis_root = root_path(root, args.analysis_root)
    run_root.mkdir(parents=True, exist_ok=True)
    analysis_root.mkdir(parents=True, exist_ok=True)
    status_path = run_root / "seed_pipeline_status.jsonl"
    write_event(
        status_path,
        "pipeline_start",
        base_config=relpath(root, base_config_path),
        seeds=[int(seed) for seed in args.seeds],
        dry_run=bool(args.dry_run),
    )

    start = time.time()
    results: list[dict[str, Any]] = []
    for seed in args.seeds:
        result = run_one_seed(args, root, base_cfg, base_config_path, int(seed))
        results.append(result)
        has_error = int(result.get("train_exit_code") or 0) != 0 or any(int(code) != 0 for code in result.get("analysis_exit_codes", {}).values())
        if has_error and args.stop_on_error:
            break

    summary = {
        "base_config": relpath(root, base_config_path),
        "elapsed_sec": time.time() - start,
        "seeds": [int(seed) for seed in args.seeds],
        "dry_run": bool(args.dry_run),
        "results": results,
    }
    (analysis_root / "seed_pipeline_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    flat_rows = []
    for result in results:
        best = result.get("best_checkpoint") or {}
        row = {
            "seed": result.get("seed"),
            "config": result.get("config"),
            "run_dir": result.get("run_dir"),
            "train_decision": result.get("train_decision"),
            "train_exit_code": result.get("train_exit_code"),
            "best_step": best.get("step"),
            "best_source": best.get("source"),
            "best_heldout_accuracy": best.get("heldout_accuracy"),
            "best_train_accuracy": best.get("train_accuracy"),
            "checkpoint_analysis_exit_code": result.get("analysis_exit_codes", {}).get("checkpoint"),
            "causal_exit_code": result.get("analysis_exit_codes", {}).get("causal"),
            "sector_cca_exit_code": result.get("analysis_exit_codes", {}).get("sector_cca"),
        }
        flat_rows.append(row)
    csv_write(analysis_root / "seed_pipeline_summary.csv", flat_rows)
    write_event(status_path, "pipeline_done", elapsed_sec=summary["elapsed_sec"], seeds_done=len(results))
    print(json.dumps({"event": "seed_pipeline_done", "summary": relpath(root, analysis_root / "seed_pipeline_summary.json")}, sort_keys=True))

    if args.stop_on_error:
        has_error = any(
            int(result.get("train_exit_code") or 0) != 0
            or any(int(code) != 0 for code in result.get("analysis_exit_codes", {}).values())
            for result in results
        )
        if has_error:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
