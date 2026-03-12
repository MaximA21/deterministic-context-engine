#!/usr/bin/env python3
"""Download AGENTbench dataset from HuggingFace.

138 real-world instances from 12 Python repos.
Source: https://huggingface.co/datasets/eth-sri/agentbench
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
DATASET_ID = "eth-sri/agentbench"


def download(force: bool = False) -> Path:
    """Download the AGENTbench dataset. Returns path to the saved JSON file."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package required. Install with: pip install datasets")
        sys.exit(1)

    output_path = DATA_DIR / "agentbench_138.json"
    if output_path.exists() and not force:
        print(f"Dataset already exists at {output_path}")
        print(f"Use --force to re-download.")
        return output_path

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {DATASET_ID} from HuggingFace...")
    ds = load_dataset(DATASET_ID, split="train")
    print(f"Downloaded {len(ds)} instances.")

    # Save as JSON for offline use
    records = [dict(row) for row in ds]
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2, default=str)
    print(f"Saved to {output_path}")

    # Print summary
    repos = {}
    for r in records:
        repo = r.get("repo") or r.get("base_repo", "unknown")
        repos[repo] = repos.get(repo, 0) + 1
    print(f"\nRepositories ({len(repos)}):")
    for repo, count in sorted(repos.items(), key=lambda x: -x[1]):
        print(f"  {repo}: {count} instances")

    return output_path


def load_instances(path: Path | None = None) -> list[dict]:
    """Load instances from local JSON. Downloads if not present."""
    path = path or DATA_DIR / "agentbench_138.json"
    if not path.exists():
        download()
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    force = "--force" in sys.argv
    download(force=force)
