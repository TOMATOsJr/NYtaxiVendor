"""Run the complete data download → Spark → statistical inference → ML workflow."""

from __future__ import annotations

import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"


def main() -> None:
    steps = [
        ("1/4 · Download and profile TLC data", "01_download_data.py"),
        ("2/4 · PySpark ETL, Spark SQL, and charts 01–03", "02_spark_etl_and_sql.py"),
        ("3/4 · Statistical inference and charts 04–06", "03_ab_statistical_inference.py"),
        ("4/4 · ML prototype and chart 07", "04_ml_tip_prototype.py"),
    ]
    for title, script in steps:
        print(f"\n{'=' * 76}\n{title}\n{'=' * 76}", flush=True)
        subprocess.run([sys.executable, str(SCRIPTS / script)], cwd=ROOT, check=True)
    print(f"\n✅ Pipeline complete. Charts: {ROOT / 'charts'}", flush=True)


if __name__ == "__main__":
    main()
