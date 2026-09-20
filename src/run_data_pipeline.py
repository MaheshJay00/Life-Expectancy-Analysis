from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
DATA_PROCESSING_DIR = SRC_DIR / "data_processing"


def run_step(script_name: str) -> None:
    script_path = DATA_PROCESSING_DIR / script_name

    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    print(f"\n=== Running {script_path.name} ===")
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
        check=False,
    )

    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    print("Starting the full data pipeline...")
    run_step("data_download.py")
    run_step("data_cleaning.py")
    print("\nData pipeline completed successfully.")


if __name__ == "__main__":
    main()
