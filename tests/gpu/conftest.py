"""Shared, dependency-light fixtures for real GPU acceptance tests.

The fixture deliberately fails when the operator did not provide an explicit
configuration. GPU tests must not silently skip and later be reported as
successful. It only validates the baseline inputs; it does not start Ray or
create any replica.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Any

import pytest


REQUIRED_KEYS = {
    "ray_address",
    "ray_namespace",
    "verl_source_root",
    "native_config",
    "model_path",
    "output_dir",
    "nodes",
    "gpus_per_node",
    "timeout_s",
}


def _read_config() -> dict[str, Any]:
    config_path = os.environ.get("MT_GPU_TEST_CONFIG")
    if not config_path:
        raise pytest.UsageError(
            "GPU acceptance tests require MT_GPU_TEST_CONFIG pointing to a JSON baseline configuration"
        )

    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise pytest.UsageError(f"MT_GPU_TEST_CONFIG does not point to a file: {path}")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise pytest.UsageError(f"Cannot read GPU baseline configuration {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise pytest.UsageError("GPU baseline configuration must be a JSON object")

    missing = sorted(REQUIRED_KEYS - config.keys())
    if missing:
        raise pytest.UsageError(f"GPU baseline configuration is missing keys: {', '.join(missing)}")

    for key in ("native_config", "model_path", "verl_source_root"):
        value = config[key]
        if not isinstance(value, str) or not value.strip():
            raise pytest.UsageError(f"GPU baseline configuration {key!r} must be a non-empty path")
        resolved = Path(value).expanduser().resolve()
        if not resolved.exists():
            raise pytest.UsageError(f"GPU baseline configuration {key!r} does not exist: {resolved}")
        config[key] = str(resolved)

    if not isinstance(config["ray_address"], str) or not config["ray_address"].strip():
        raise pytest.UsageError("GPU baseline configuration ray_address must be 'local' or a non-empty Ray address")
    if not isinstance(config["ray_namespace"], str) or not config["ray_namespace"].strip():
        raise pytest.UsageError("GPU baseline configuration ray_namespace must be non-empty")

    for key in ("nodes", "gpus_per_node"):
        value = config[key]
        if type(value) is not int or value <= 0:
            raise pytest.UsageError(f"GPU baseline configuration {key} must be a positive integer")
    if type(config["timeout_s"]) not in (int, float) or config["timeout_s"] <= 0:
        raise pytest.UsageError("GPU baseline configuration timeout_s must be positive")

    if not isinstance(config["output_dir"], str) or not config["output_dir"].strip():
        raise pytest.UsageError("GPU baseline configuration output_dir must be a non-empty path")
    output_dir = Path(config["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config["output_dir"] = str(output_dir)
    config["config_path"] = str(path)
    return config


@pytest.fixture(scope="session")
def gpu_test_config() -> dict[str, Any]:
    """Return the validated operator-provided GPU acceptance configuration."""

    return _read_config()


@pytest.fixture(scope="session")
def gpu_inventory(gpu_test_config: dict[str, Any]) -> list[str]:
    """Return visible GPU UUIDs and fail before any Ray/replica creation."""

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=float(gpu_test_config["timeout_s"]),
        )
    except FileNotFoundError as exc:
        raise pytest.UsageError("GPU acceptance tests require nvidia-smi on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise pytest.UsageError(f"nvidia-smi failed before Ray startup: {exc.stderr.strip()}") from exc
    except subprocess.TimeoutExpired as exc:
        raise pytest.UsageError("nvidia-smi timed out before Ray startup") from exc

    uuids = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    # nvidia-smi reports the node running this test. A multi-node baseline
    # checks gpus_per_node here; the Ray/GPU acceptance run checks every node.
    required = gpu_test_config["gpus_per_node"]
    if len(uuids) < required:
        raise pytest.UsageError(f"Need at least {required} visible GPUs, found {len(uuids)}")
    return uuids
