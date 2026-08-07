from __future__ import annotations

from pathlib import Path

from .config import load_config
from .train import train


def main() -> None:
    root = Path(__file__).resolve().parent
    configs = [
        root / "configs" / "smoke_three_sector.yaml",
        root / "configs" / "smoke_ordered_route.yaml",
    ]
    for config_path in configs:
        print(f"running {config_path}")
        train(load_config(config_path))


if __name__ == "__main__":
    main()
