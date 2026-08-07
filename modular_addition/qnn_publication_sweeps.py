from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

import yaml


ARCHITECTURES = {
    "aux": {
        "readout_type": "layerwise_dirac_aux",
        "lr": 0.003,
        "layerwise_dirac_loss_weight": 0.50,
        "fourier_auxiliary_weight": 0.30,
        "dirac_coefficient_norm_weight": 0.002,
        "adapter_scale": None,
    },
    "adapter": {
        "readout_type": "layerwise_dirac_adapter",
        "lr": 0.002,
        "layerwise_dirac_loss_weight": 0.50,
        "fourier_auxiliary_weight": 0.30,
        "dirac_coefficient_norm_weight": 0.002,
        "adapter_scale": 0.20,
    },
    "residual": {
        "readout_type": "layerwise_dirac_residual",
        "lr": 0.003,
        "layerwise_dirac_loss_weight": 0.35,
        "fourier_auxiliary_weight": 0.0,
        "dirac_coefficient_norm_weight": 0.0,
        "adapter_scale": None,
    },
    "mean": {
        "readout_type": "layerwise_dirac_mean",
        "lr": 0.003,
        "layerwise_dirac_loss_weight": 0.50,
        "fourier_auxiliary_weight": 0.0,
        "dirac_coefficient_norm_weight": 0.0,
        "adapter_scale": None,
    },
    "ensemble": {
        "readout_type": "layerwise_dirac_ensemble",
        "lr": 0.003,
        "layerwise_dirac_loss_weight": 0.35,
        "fourier_auxiliary_weight": 0.0,
        "dirac_coefficient_norm_weight": 0.0,
        "adapter_scale": None,
    },
}


CLASSICAL_BASELINES = {
    "fourier_linear": {"readout_type": "linear", "hidden_dim": 0, "target_params": None, "feature_mode": "separate"},
    "fourier_mlp_matched": {"readout_type": "linear", "hidden_dim": "auto", "target_params": 23428, "feature_mode": "separate"},
    "fourier_delta_matched": {"readout_type": "fourier_delta", "hidden_dim": "auto", "target_params": 23428, "feature_mode": "separate"},
    "fourier_product_linear": {"readout_type": "linear", "hidden_dim": 0, "target_params": None, "feature_mode": "product"},
    "fourier_product_delta": {"readout_type": "fourier_delta", "hidden_dim": 0, "target_params": None, "feature_mode": "product"},
    "raw_numeric_mlp_matched": {"readout_type": "linear", "hidden_dim": "auto", "target_params": 23428, "feature_mode": "raw"},
}


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def fraction_tag(value: float) -> str:
    return f"train{int(round(float(value) * 100)):02d}"


def qbits_for_modulus(modulus: int) -> int:
    return int(math.ceil(math.log2(modulus)))


def max_frequency(modulus: int) -> int:
    return min(21, modulus // 2)


def boundary_width(modulus: int, default: int = 5) -> int:
    return max(2, min(default, modulus // 16 + 2))


def qnn_steps(modulus: int) -> int:
    if modulus <= 31:
        return 2000
    if modulus <= 97:
        return 2500
    return 3000


def qnn_config(
    modulus: int,
    seed: int,
    arch: str,
    *,
    device: str,
    train_fraction: float,
    split_seed: int = 0,
) -> dict[str, Any]:
    arch_cfg = ARCHITECTURES[arch]
    freq = max_frequency(modulus)
    if split_seed == 0 and abs(float(train_fraction) - 0.30) < 1.0e-9:
        run_name = f"modular_addition_qnn_pub_mod{modulus}_{arch}_seed{seed}"
    else:
        run_name = f"modular_addition_qnn_pub_mod{modulus}_{arch}_{fraction_tag(train_fraction)}_split{split_seed}_seed{seed}"
    model_seed = 12000 + 100 * modulus + 10 * list(ARCHITECTURES).index(arch) + seed
    model: dict[str, Any] = {
        "n_qubits": qbits_for_modulus(modulus),
        "n_layers": 4,
        "readout_type": arch_cfg["readout_type"],
        "fourier_max_frequency": freq,
        "fourier_kernel_init": "fejer",
        "fourier_residual_linear": False,
        "fourier_residual_scale": 0.0,
        "dirac_coefficient_mode": "none",
        "dirac_coefficient_eps": 1.0e-6,
        "dirac_kernel_trainable": True,
        "auxiliary_head_moduli": [],
    }
    if arch_cfg["adapter_scale"] is not None:
        model["layerwise_dirac_adapter_scale"] = arch_cfg["adapter_scale"]
    training: dict[str, Any] = {
        "batch_size": 256,
        "eval_batch_size": 512,
        "steps": qnn_steps(modulus),
        "lr": arch_cfg["lr"],
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "eval_every": 250,
        "max_eval_batches": None,
        "disable_progress": True,
        "direct_auxiliary_heads": False,
        "auxiliary_residue_losses": [],
        "layerwise_dirac_loss_weight": arch_cfg["layerwise_dirac_loss_weight"],
        "layerwise_dirac_depth_weighted": True,
        "fourier_auxiliary_weight": arch_cfg["fourier_auxiliary_weight"],
        "dirac_coefficient_norm_weight": arch_cfg["dirac_coefficient_norm_weight"],
        "hard_neighbor_margin_weight": 0.35,
        "hard_neighbor_margin": 3.0,
        "hard_neighbor_offsets": [1, 2],
        "hard_neighbor_focus_gamma": 1.0,
        "boundary_oversample_weight": 8.0,
        "boundary_operand_width": boundary_width(modulus),
        "boundary_residue_width": boundary_width(modulus, 4),
        "wrap_boundary_width": boundary_width(modulus),
        "initial_readout_refresh_steps": 0,
        "readout_refresh_every": 0,
        "readout_refresh_steps": 0,
        "final_readout_refresh_steps": 0,
        "early_stopping_metric": "test_accuracy",
        "early_stopping_mode": "max",
        "early_stopping_min_delta": 5.0e-5,
        "early_stopping_min_step": 750,
        "early_stopping_patience_evals": 5,
        "early_stopping_restore_best": True,
        "run_final_refresh_on_early_stop": False,
    }
    if arch in {"aux", "adapter"}:
        training.update(
            {
                "fourier_auxiliary_min_frequency": 1,
                "fourier_auxiliary_max_frequency": freq,
                "fourier_auxiliary_normalize_pairs": True,
                "fourier_auxiliary_frequency_power": 1.0,
                "dirac_coefficient_norm_target": 1.0,
            }
        )
    return {
        "seed": model_seed,
        "device": device,
        "output_dir": f"runs/{run_name}",
        "dataset": {"modulus": modulus, "train_fraction": train_fraction, "seed": split_seed},
        "model": model,
        "training": training,
        "notes": {
            "publication_sweep": True,
            "architecture": arch,
            "model_seed_index": seed,
            "split_seed": split_seed,
        },
    }


def classical_config(
    modulus: int,
    seed: int,
    baseline: str,
    *,
    device: str,
    train_fraction: float,
    split_seed: int = 0,
) -> dict[str, Any]:
    base = CLASSICAL_BASELINES[baseline]
    if split_seed == 0 and abs(float(train_fraction) - 0.30) < 1.0e-9:
        run_name = f"modular_addition_classical_pub_mod{modulus}_{baseline}_seed{seed}"
    else:
        run_name = f"modular_addition_classical_pub_mod{modulus}_{baseline}_{fraction_tag(train_fraction)}_split{split_seed}_seed{seed}"
    return {
        "seed": 22000 + 100 * modulus + 10 * list(CLASSICAL_BASELINES).index(baseline) + seed,
        "device": device,
        "output_dir": f"runs/{run_name}",
        "dataset": {"modulus": modulus, "train_fraction": train_fraction, "seed": split_seed},
        "model": {
            "input_max_frequency": max_frequency(modulus),
            "hidden_dim": base["hidden_dim"],
            "hidden_layers": 1 if base["hidden_dim"] != 0 else 0,
            "readout_type": base["readout_type"],
            "feature_mode": base["feature_mode"],
            "fourier_max_frequency": max_frequency(modulus),
            "target_params": base["target_params"],
        },
        "training": {
            "steps": 2500 if modulus <= 97 else 3000,
            "batch_size": 256,
            "eval_batch_size": 1024,
            "lr": 0.003,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "eval_every": 250,
            "early_stopping_patience_evals": 5,
            "early_stopping_min_step": 750,
            "early_stopping_min_delta": 5.0e-5,
            "early_stopping_restore_best": True,
            "disable_progress": True,
        },
        "notes": {
            "publication_sweep": True,
            "baseline": baseline,
            "model_seed_index": seed,
            "split_seed": split_seed,
        },
    }


def write_yaml(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def summarize_existing_run(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {"run_dir": str(run_dir), "status": "missing"}
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"run_dir": str(run_dir), "status": "bad_json"}
    variants = data.get("variants")
    if isinstance(variants, list) and variants:
        row = variants[0]
        best = row.get("best_record", {})
        return {
            "run_dir": str(run_dir),
            "status": "complete",
            "best_test_accuracy": row.get("best_test_accuracy"),
            "final_test_accuracy": row.get("final_test_accuracy"),
            "final_train_accuracy": row.get("final_train_accuracy"),
            "best_step": best.get("step"),
            "parameter_count": row.get("parameter_count"),
        }
    if "best_test_accuracy" in data:
        return {
            "run_dir": str(run_dir),
            "status": "complete",
            "best_test_accuracy": data.get("best_test_accuracy"),
            "final_test_accuracy": data.get("final_test_accuracy"),
            "final_train_accuracy": data.get("final_train_accuracy"),
            "best_step": data.get("best_step"),
            "parameter_count": data.get("parameter_count"),
        }
    return {"run_dir": str(run_dir), "status": "unknown_schema"}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fmt_float(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return ""


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, float, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "complete":
            continue
        key = (
            str(row.get("kind")),
            int(row.get("modulus")),
            float(row.get("train_fraction", 0.30)),
            str(row.get("architecture")),
        )
        grouped.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for (kind, modulus, train_fraction, architecture), group in sorted(grouped.items(), key=lambda x: (x[0][1], x[0][2], x[0][0], x[0][3])):
        held = [float(row["best_test_accuracy"]) for row in group if row.get("best_test_accuracy") not in {None, ""}]
        train = [float(row["final_train_accuracy"]) for row in group if row.get("final_train_accuracy") not in {None, ""}]
        params = [float(row["parameter_count"]) for row in group if row.get("parameter_count") not in {None, ""}]
        split_seeds = sorted({int(row.get("split_seed", 0)) for row in group})
        model_seeds = sorted({int(row.get("seed", 0)) for row in group})
        out.append(
            {
                "kind": kind,
                "modulus": modulus,
                "train_fraction": train_fraction,
                "architecture": architecture,
                "runs_completed": len(group),
                "model_seeds": ",".join(str(x) for x in model_seeds),
                "split_seeds": ",".join(str(x) for x in split_seeds),
                "heldout_mean": statistics.mean(held) if held else None,
                "heldout_std": statistics.stdev(held) if len(held) > 1 else 0.0 if held else None,
                "heldout_min": min(held) if held else None,
                "heldout_max": max(held) if held else None,
                "train_mean": statistics.mean(train) if train else None,
                "parameter_count_mean": statistics.mean(params) if params else None,
            }
        )
    return out


def write_aggregate_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# QNN Publication Sweep Aggregate",
        "",
        "Mean and sample standard deviation are computed across completed runs in each modulus/train-fraction/family group.",
        "",
        "| kind | modulus | train fraction | architecture | runs | model seeds | split seeds | held-out mean | held-out std | held-out min | held-out max | train mean | params |",
        "| --- | ---: | ---: | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['kind']} | {row['modulus']} | {fmt_float(row['train_fraction'])} | {row['architecture']} | "
            f"{row['runs_completed']} | {row['model_seeds']} | {row['split_seeds']} | "
            f"{fmt_float(row['heldout_mean'])} | {fmt_float(row['heldout_std'])} | "
            f"{fmt_float(row['heldout_min'])} | {fmt_float(row['heldout_max'])} | "
            f"{fmt_float(row['train_mean'])} | {fmt_float(row['parameter_count_mean'])} |"
        )
    lines.extend(
        [
            "",
            "Interpretation guardrail: `fourier_product_delta` is an upper-bound sanity check, not a fair learned baseline, because it receives explicit product/addition Fourier features.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def generate(args: argparse.Namespace) -> None:
    moduli = parse_ints(args.moduli)
    seeds = parse_ints(args.seeds)
    split_seeds = parse_ints(args.split_seeds)
    train_fractions = parse_floats(args.train_fractions if args.train_fractions else str(args.train_fraction))
    archs = [x.strip() for x in args.architectures.split(",") if x.strip()]
    baselines = [] if getattr(args, "no_baselines", False) else [x.strip() for x in args.baselines.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    commands: list[str] = []
    rows: list[dict[str, Any]] = []

    for modulus in moduli:
        for train_fraction in train_fractions:
            frac_tag = "" if abs(float(train_fraction) - 0.30) < 1.0e-9 else f"_{fraction_tag(train_fraction)}"
            for split_seed in split_seeds:
                split_tag = "" if split_seed == 0 and abs(float(train_fraction) - 0.30) < 1.0e-9 else f"_split{split_seed}"
                for seed in seeds:
                    for arch in archs:
                        cfg = qnn_config(modulus, seed, arch, device=args.device, train_fraction=train_fraction, split_seed=split_seed)
                        cfg_path = Path("configs") / f"modular_addition_qnn_pub_mod{modulus}_{arch}{frac_tag}{split_tag}_seed{seed}.yaml"
                        if args.write_configs:
                            write_yaml(cfg_path, cfg)
                        cmd = f"python -m modular_addition.qnn_mod97 --config {cfg_path} --variants prob_head --device {args.device} --disable-progress"
                        commands.append(cmd)
                        status = summarize_existing_run(Path(cfg["output_dir"]))
                        rows.append(
                            {
                                "kind": "qnn",
                                "modulus": modulus,
                                "train_fraction": train_fraction,
                                "split_seed": split_seed,
                                "seed": seed,
                                "architecture": arch,
                                "config": str(cfg_path),
                                "command": cmd,
                                **status,
                            }
                        )
                    for baseline in baselines:
                        cfg = classical_config(modulus, seed, baseline, device=args.device, train_fraction=train_fraction, split_seed=split_seed)
                        cfg_path = Path("configs") / f"modular_addition_classical_pub_mod{modulus}_{baseline}{frac_tag}{split_tag}_seed{seed}.yaml"
                        if args.write_configs:
                            write_yaml(cfg_path, cfg)
                        cmd = f"python -m modular_addition.classical_fourier_baseline --config {cfg_path} --device {args.device}"
                        commands.append(cmd)
                        status = summarize_existing_run(Path(cfg["output_dir"]))
                        rows.append(
                            {
                                "kind": "classical",
                                "modulus": modulus,
                                "train_fraction": train_fraction,
                                "split_seed": split_seed,
                                "seed": seed,
                                "architecture": baseline,
                                "config": str(cfg_path),
                                "command": cmd,
                                **status,
                            }
                        )

    manifest_lines = [
        "# QNN Publication Sweep Command Manifest",
        "",
        f"Moduli: `{moduli}`",
        f"Model seeds: `{seeds}`",
        f"Split seeds: `{split_seeds}`",
        f"Train fractions: `{train_fractions}`",
        "",
        "Run commands one at a time or in a scheduler. Configs record model seed, data split seed, and train fraction explicitly.",
        "",
        "```powershell",
        *commands,
        "```",
        "",
    ]
    (out_dir / "COMMAND_MANIFEST.md").write_text("\n".join(manifest_lines), encoding="utf-8")
    write_csv(out_dir / "sweep_status.csv", rows)
    aggregates = aggregate_rows(rows)
    write_csv(out_dir / "sweep_aggregate.csv", aggregates)
    write_aggregate_markdown(out_dir / "SWEEP_AGGREGATE.md", aggregates)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate publication-readiness seed/modulus sweep configs and manifests.")
    parser.add_argument("--moduli", default="31,97,127")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--architectures", default="aux,adapter,residual")
    parser.add_argument("--baselines", default="fourier_linear,fourier_mlp_matched,fourier_delta_matched,fourier_product_linear,fourier_product_delta,raw_numeric_mlp_matched")
    parser.add_argument("--no-baselines", action="store_true")
    parser.add_argument("--train-fraction", type=float, default=0.30)
    parser.add_argument("--train-fractions", default=None)
    parser.add_argument("--split-seeds", default="0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default="analysis/qnn_publication_sweep")
    parser.add_argument("--write-configs", action="store_true")
    return parser.parse_args()


def main() -> None:
    generate(parse_args())


if __name__ == "__main__":
    main()
