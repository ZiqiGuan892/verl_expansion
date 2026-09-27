"""Dependency-light acceptance deployment/configuration checks before Ray starts."""

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path


def verify_config(config, environ=None):
    """Reject a requested E2E run that actually selected a legacy smoke.

    Called both by the driver profile resolver and by TaskRunner before GS or
    GPU initialization. Normal jobs without the opt-in flag are unchanged.
    """
    env = os.environ if environ is None else environ
    get = getattr(config, "get", lambda key, default=None: default)
    multitask = get("multitask", {}) or {}
    if not hasattr(multitask, "get"):
        multitask = {}
    e2e = multitask.get("e2e_test", {}) or {}
    active = hasattr(e2e, "get") and e2e.get("enabled", False) is True
    if env.get("MULTITASK_E2E_REQUIRED") != "1" and not active:
        return None
    if not active:
        raise RuntimeError(
            "D0_D4_E2E_CONFIG_MISMATCH: E2E requested but multitask.e2e_test.enabled is not true. "
            "Use D4_test.sh from VERL_MULTI_TASK_ROOT and ensure multi_task_run.sh forwards all arguments."
        )
    for name in ("d2_runtime_test", "d3_bootstrap_test", "d4_runtime_test"):
        if (multitask.get(name, {}) or {}).get("enabled", False):
            raise RuntimeError(f"D0_D4_E2E_CONFIG_MISMATCH: legacy {name} must be disabled")
    from .e2e_verdict import SCENARIOS

    scenario = e2e.get("scenario", "basic")
    expected = env.get("MULTITASK_E2E_SCENARIO", scenario)
    if scenario not in SCENARIOS or scenario != expected:
        raise RuntimeError(f"D0_D4_E2E_CONFIG_MISMATCH: expected scenario={expected}, got {scenario}")
    source = Path(__file__).resolve().parents[3]
    expected_root = env.get("MULTITASK_E2E_SOURCE_ROOT")
    if expected_root and source != Path(expected_root).resolve():
        raise RuntimeError(f"D0_D4_E2E_SOURCE_MISMATCH: imported {source}, expected {expected_root}")
    profile = (multitask.get("runtime", {}) or {}).get("profile")
    if profile != "experimental_fully_async_standalone":
        raise RuntimeError("D0_D4_E2E_CONFIG_MISMATCH: multi-task runtime profile must be enabled")
    return {"scenario": scenario, "source_root": str(source), "legacy_fixtures_disabled": True}


def check_deployment(source_root, fixture_script, launcher):
    """Check the actual fixture copy and imported package; no Ray/vLLM import."""
    root = Path(source_root).resolve()
    script = Path(fixture_script).resolve()
    launch = Path(launcher).resolve()
    canonical = root / "D4_test.sh"
    if script.read_text(encoding="utf-8") != canonical.read_text(encoding="utf-8"):
        raise RuntimeError(f"stale D4 fixture {script}; use {canonical}")
    if not launch.is_file():
        raise RuntimeError(f"missing training launcher: {launch}")
    package = importlib.util.find_spec("multi_task_scheduler")
    expected = root / "src/multi_task_scheduler/__init__.py"
    if package is None or not package.origin or Path(package.origin).resolve() != expected:
        raise RuntimeError(f"plugin import mismatch: {getattr(package, 'origin', None)}; expected {expected}")
    hooks = {
        "task_runner": ("_run_training_loop", "run_training_fixture"),
        "rollouter": ("e2e_test_action", "dispatch_rollout"),
        "trainer": ("e2e_test_action", "dispatch_trainer"),
    }
    hashes = {"D4_test.sh": hashlib.sha256(canonical.read_bytes()).hexdigest()}
    for name, (method, callee) in hooks.items():
        path = root / f"src/multi_task_scheduler/integration/verl/experimental_fully_async/{name}.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        methods = [node for node in ast.walk(tree)
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method]
        if not any(isinstance(node, ast.Name) and node.id == callee for item in methods for node in ast.walk(item)):
            raise RuntimeError(f"missing E2E hook in {path}; update the complete plugin checkout")
        hashes[f"{name}.py"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"source_root": str(root), "fixture": str(script), "launcher": str(launch), "sha256": hashes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--fixture-script", required=True)
    parser.add_argument("--launcher", required=True)
    args = parser.parse_args()
    result = check_deployment(args.source_root, args.fixture_script, args.launcher)
    print("D0_D4_E2E_PREFLIGHT " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
