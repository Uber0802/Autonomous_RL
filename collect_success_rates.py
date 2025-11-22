#!/usr/bin/env python3
import yaml
import re
import sys
from pathlib import Path

def find_glob_dir(run_dir: Path):
    """Search recursively for the first directory named 'glob'."""
    for p in run_dir.rglob("glob"):
        if p.is_dir():
            return p
    return None

def collect_success_rates(run_paths):
    for run_path in run_paths:
        run = Path(run_path).resolve()
        print(f"\n=== {run.name} ===")
        glob_path = find_glob_dir(run)

        if not glob_path:
            print(f"⚠️  No 'glob' directory found inside {run}")
            continue

        subdirs = sorted(glob_path.iterdir(), key=lambda x: tuple(map(int, re.findall(r'\d+', x.name))))
        grouped = {}

        for subdir in subdirs:
            stats_file = subdir / "stats.yaml"
            if not stats_file.exists():
                continue
            with open(stats_file, "r") as f:
                data = yaml.safe_load(f) or {}
            rate = data.get("stats", {}).get("success", {})
            if isinstance(rate, dict):
                rate = rate.get("success_rate")
            prefix = subdir.name.split('-')[0]
            grouped.setdefault(prefix, []).append(rate)

        for prefix, rates in grouped.items():
            if rates:
                start = f"{prefix}-0"
                end = f"{prefix}-{len(rates)-1}"
                print(f"{start} to {end}")
                for r in rates:
                    print(f"{r:.4f}")
                #print("\n")

if __name__ == "__main__":
    # Run paths passed as command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python collect_success_rates.py <run_path1> <run_path2> ...")
        sys.exit(1)
    collect_success_rates(sys.argv[1:])