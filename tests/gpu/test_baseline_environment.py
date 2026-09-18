"""D0 baseline checks; these do not create Ray actors or replicas."""

import json
from pathlib import Path

import pytest


pytestmark = pytest.mark.gpu_integration


def test_gpu_baseline_inputs_are_explicit(gpu_test_config: dict, gpu_inventory: list[str]):
    """Verify the selected native configuration and visible devices before startup."""

    assert Path(gpu_test_config["verl_source_root"]).is_dir()
    assert Path(gpu_test_config["native_config"]).is_file()
    assert Path(gpu_test_config["model_path"]).exists()
    assert len(gpu_inventory) >= gpu_test_config["gpus_per_node"]


def test_gpu_baseline_record_is_serializable(gpu_test_config: dict, tmp_path: Path):
    """Ensure the baseline metadata can be archived with a run result."""

    record_path = tmp_path / "baseline-config.json"
    record_path.write_text(json.dumps(gpu_test_config, indent=2, sort_keys=True), encoding="utf-8")
    assert json.loads(record_path.read_text(encoding="utf-8")) == gpu_test_config
