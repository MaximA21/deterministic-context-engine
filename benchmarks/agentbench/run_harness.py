#!/usr/bin/env python3
"""Full Docker-based AGENTbench evaluation harness.

Runs our context engine as the context management layer for a coding agent,
then evaluates patches using AGENTbench's two-tier test protocol:
  1. Instance-specific tests (does the patch solve the issue?)
  2. Repo regression tests (does the patch break anything?)

Requires: Docker, API key for the coding model.

Compatible with AGENTbench's preds.json / report.json format so results
can be directly compared with their published numbers.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.agentbench.download import load_instances
from benchmarks.agentbench.diff_filter import filter_patch

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
CONTAINER_TIMEOUT = 1800  # 30 min per instance
WORKDIR = "/project/testbed"


@dataclass
class EvalResult:
    instance_id: str
    resolved: bool
    instance_test_passed: bool
    repo_test_passed: bool
    model_patch: str
    error: str | None = None


def _docker_exec(container_id: str, cmd: str, timeout: int = 300) -> tuple[int, str]:
    """Run a command inside a Docker container."""
    result = subprocess.run(
        ["docker", "exec", container_id, "bash", "-c", cmd],
        capture_output=True, text=True, timeout=timeout,
    )
    output = result.stdout + result.stderr
    return result.returncode, output


def _docker_cp_to(container_id: str, local_path: str, remote_path: str) -> None:
    """Copy a file into a container."""
    subprocess.run(
        ["docker", "cp", local_path, f"{container_id}:{remote_path}"],
        check=True, capture_output=True,
    )


def _start_container(docker_image: str) -> str:
    """Start a Docker container and return its ID."""
    result = subprocess.run(
        [
            "docker", "run", "-d", "--rm", "--network=host",
            docker_image, "sleep", "7200",
        ],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def _stop_container(container_id: str) -> None:
    """Stop and remove a container."""
    subprocess.run(
        ["docker", "stop", container_id],
        capture_output=True, timeout=30,
    )


def setup_container(container_id: str, instance: dict) -> bool:
    """Prepare a container for evaluation: checkout base SHA, run setup."""
    base_sha = instance.get("base_sha", "")

    # Move testbed if needed (AGENTbench convention)
    _docker_exec(container_id, f"[ -d /testbed ] && mv /testbed {WORKDIR} || true")
    _docker_exec(container_id, f"mkdir -p {WORKDIR}")

    # Checkout the base SHA
    if base_sha:
        rc, out = _docker_exec(
            container_id,
            f"cd {WORKDIR} && git checkout -f {base_sha}",
        )
        if rc != 0:
            print(f"  WARNING: git checkout failed: {out[:200]}")

    # Run setup commands
    setup_commands = instance.get("setup_commands") or []
    for cmd in setup_commands:
        rc, out = _docker_exec(container_id, f"cd {WORKDIR} && {cmd}", timeout=600)
        if rc != 0:
            print(f"  WARNING: setup command failed: {cmd[:80]} -> {out[:200]}")

    return True


def apply_patch(container_id: str, patch: str) -> tuple[bool, str]:
    """Apply a model patch inside the container."""
    if not patch or not patch.strip():
        return False, "Empty patch"

    # Filter patch (AGENTbench rules: Python only, no venvs, no root tests)
    filtered = filter_patch(patch)
    if not filtered.cleaned.strip():
        return False, f"Patch filtered to nothing (dropped: {filtered.files_dropped})"

    # Write patch to temp file, copy into container
    with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
        f.write(filtered.cleaned)
        patch_path = f.name

    try:
        _docker_cp_to(container_id, patch_path, f"{WORKDIR}/model.patch")
        rc, out = _docker_exec(
            container_id,
            f"cd {WORKDIR} && git apply --whitespace=nowarn model.patch",
        )
        if rc != 0:
            return False, f"git apply failed: {out[:500]}"
        return True, ""
    finally:
        os.unlink(patch_path)


def run_instance_tests(container_id: str, instance: dict) -> tuple[bool, dict]:
    """Run instance-specific tests. Returns (passed, test_results)."""
    test_names = instance.get("test_file_names") or []
    test_contents = instance.get("test_file_contents") or []
    test_runner = instance.get("test_file_runner", "")
    test_commands = instance.get("test_commands") or []

    if not test_commands:
        return True, {"skipped": True}

    # Write test files
    for name, content in zip(test_names, test_contents):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            _docker_cp_to(container_id, f.name, f"{WORKDIR}/{name}")
            os.unlink(f.name)

    # Write test runner
    if test_runner:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_runner)
            _docker_cp_to(container_id, f.name, f"{WORKDIR}/run_pr_tests.py")
            os.unlink(f.name)

    # Execute tests
    for cmd in test_commands:
        rc, out = _docker_exec(
            container_id,
            f"cd {WORKDIR} && {cmd}",
            timeout=CONTAINER_TIMEOUT,
        )

    # Read results
    rc, out = _docker_exec(
        container_id,
        f"cat {WORKDIR}/pr_test_results.json 2>/dev/null || echo '{{}}'",
    )
    try:
        results = json.loads(out.strip())
    except json.JSONDecodeError:
        results = {"raw_output": out[:2000]}

    # Check: all tests must pass
    if isinstance(results, dict):
        passed = all(v is True for v in results.values()) if results else rc == 0
    else:
        passed = rc == 0

    return passed, results


def run_repo_tests(container_id: str, instance: dict) -> tuple[bool, dict]:
    """Run repo-wide regression tests. Returns (passed, test_results)."""
    test_commands = instance.get("repo_test_commands") or []
    test_runner = instance.get("repo_test_runner", "")
    baseline = instance.get("repo_test_after_pr_patch") or {}

    if not test_commands:
        return True, {"skipped": True}

    # Write test runner
    if test_runner:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_runner)
            _docker_cp_to(container_id, f.name, f"{WORKDIR}/run_tests.py")
            os.unlink(f.name)

    # Execute
    for cmd in test_commands:
        _docker_exec(container_id, f"cd {WORKDIR} && {cmd}", timeout=CONTAINER_TIMEOUT)

    # Read results
    rc, out = _docker_exec(
        container_id,
        f"cat {WORKDIR}/test_results.json 2>/dev/null || echo '{{}}'",
    )
    try:
        results = json.loads(out.strip())
    except json.JSONDecodeError:
        results = {}

    # Regression check: any test that passed with gold patch must still pass
    if not baseline:
        return True, results

    for test_name, gold_passed in baseline.items():
        if gold_passed and not results.get(test_name, False):
            return False, results

    return True, results


def evaluate_patch(instance: dict, model_patch: str) -> EvalResult:
    """Evaluate a single patch against an AGENTbench instance.

    Full Docker-based evaluation with two-tier testing:
      1. Instance tests (does it solve the problem?)
      2. Repo regression tests (does it break anything?)
    """
    instance_id = instance.get("instance_id", "unknown")
    docker_image = instance.get("docker_image", "")

    if not docker_image:
        return EvalResult(
            instance_id=instance_id, resolved=False,
            instance_test_passed=False, repo_test_passed=False,
            model_patch=model_patch, error="No docker_image specified",
        )

    container_id = None
    try:
        # Start container
        container_id = _start_container(docker_image)
        setup_container(container_id, instance)

        # Apply patch
        applied, err = apply_patch(container_id, model_patch)
        if not applied:
            return EvalResult(
                instance_id=instance_id, resolved=False,
                instance_test_passed=False, repo_test_passed=False,
                model_patch=model_patch, error=f"Patch apply failed: {err}",
            )

        # Run tests
        inst_passed, inst_results = run_instance_tests(container_id, instance)
        repo_passed, repo_results = run_repo_tests(container_id, instance)

        return EvalResult(
            instance_id=instance_id,
            resolved=inst_passed and repo_passed,
            instance_test_passed=inst_passed,
            repo_test_passed=repo_passed,
            model_patch=model_patch,
        )

    except Exception as e:
        return EvalResult(
            instance_id=instance_id, resolved=False,
            instance_test_passed=False, repo_test_passed=False,
            model_patch=model_patch, error=str(e),
        )
    finally:
        if container_id:
            _stop_container(container_id)


def evaluate_preds(
    preds_path: str | Path,
    max_workers: int = 4,
    instance_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate predictions from a preds.json file.

    Compatible with AGENTbench's preds.json format:
    {
      "<instance_id>": {
        "model_patch": "<git diff>",
        "instance_id": "<id>",
        "model_name_or_path": "<model>"
      }
    }
    """
    with open(preds_path) as f:
        preds = json.load(f)

    instances = load_instances()
    instance_map = {i["instance_id"]: i for i in instances}

    if instance_ids:
        pred_ids = set(instance_ids) & set(preds.keys())
    else:
        pred_ids = set(preds.keys())

    print(f"Evaluating {len(pred_ids)} predictions...")

    report = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for iid in pred_ids:
            if iid not in instance_map:
                print(f"  SKIP {iid}: not in dataset")
                continue
            patch = preds[iid].get("model_patch", "")
            future = pool.submit(evaluate_patch, instance_map[iid], patch)
            futures[future] = iid

        for future in as_completed(futures):
            iid = futures[future]
            try:
                result = future.result(timeout=CONTAINER_TIMEOUT + 60)
                status = "PASS" if result.resolved else "FAIL"
                error_msg = f" ({result.error})" if result.error else ""
                print(f"  {status}: {iid}{error_msg}")
                report[iid] = {
                    "resolved": result.resolved,
                    "instance_test_passed": result.instance_test_passed,
                    "repo_test_passed": result.repo_test_passed,
                    "model_patch": result.model_patch[:500],
                    "error": result.error,
                }
            except Exception as e:
                print(f"  ERROR: {iid}: {e}")
                report[iid] = {
                    "resolved": False,
                    "instance_test_passed": False,
                    "repo_test_passed": False,
                    "error": str(e),
                }

    # Summary
    total = len(report)
    resolved = sum(1 for r in report.values() if r["resolved"])
    print(f"\nResolved: {resolved}/{total} ({100*resolved/total:.1f}%)" if total else "No results")

    return {"report": report, "resolved": resolved, "total": total}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="AGENTbench Docker evaluation harness")
    parser.add_argument("preds", type=str, help="Path to preds.json")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--instances", nargs="*", default=None, help="Instance IDs to evaluate")
    parser.add_argument("--output", type=str, default=None, help="Output report path")
    args = parser.parse_args()

    results = evaluate_preds(args.preds, args.workers, args.instances)

    out_path = args.output or str(OUTPUT_DIR / "report.json")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Report saved to {out_path}")


if __name__ == "__main__":
    main()
