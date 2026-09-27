"""Deployment/argument regression tests; no real Ray, training or NPU execution."""

import ast
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from multi_task_scheduler.testing import launch_check, startup_diagnostics


ROOT = Path(__file__).resolve().parents[2]


def _config():
    return {"multitask": {"runtime": {"profile": "experimental_fully_async_standalone"},
                          "e2e_test": {"enabled": True, "scenario": "split"}}}


def test_ordinary_jobs_unchanged():
    assert launch_check.verify_config({}, {}) is None
    assert launch_check.verify_config(object(), {}) is None


def test_e2e_guard_rejects_dropped_arguments_before_initialization():
    old_config = {"multitask": {"d4_runtime_test": {"enabled": True}}}
    with pytest.raises(RuntimeError, match="E2E_CONFIG_MISMATCH"):
        launch_check.verify_config(old_config, {"MULTITASK_E2E_REQUIRED": "1"})


@pytest.mark.parametrize("legacy", ["d2_runtime_test", "d3_bootstrap_test", "d4_runtime_test"])
def test_legacy_fixture_cannot_run_with_e2e(legacy):
    config = _config()
    config["multitask"][legacy] = {"enabled": True}
    with pytest.raises(RuntimeError, match="must be disabled"):
        launch_check.verify_config(config, {})


def test_scenario_source_and_profile_must_match(tmp_path):
    assert launch_check.verify_config(_config(), {"MULTITASK_E2E_SOURCE_ROOT": str(ROOT)})["scenario"] == "split"
    with pytest.raises(RuntimeError, match="expected scenario"):
        launch_check.verify_config(_config(), {"MULTITASK_E2E_SCENARIO": "basic"})
    with pytest.raises(RuntimeError, match="SOURCE_MISMATCH"):
        launch_check.verify_config(_config(), {"MULTITASK_E2E_SOURCE_ROOT": str(tmp_path)})
    config = _config()
    config["multitask"]["runtime"]["profile"] = None
    with pytest.raises(RuntimeError, match="profile must be enabled"):
        launch_check.verify_config(config, {})


def test_profile_resolver_rejects_wrong_fixture_before_ray_import(monkeypatch):
    from multi_task_scheduler.integration.verl.runtime_profile import resolve_runtime_profile

    monkeypatch.setenv("MULTITASK_E2E_REQUIRED", "1")
    before = set(sys.modules)
    with pytest.raises(RuntimeError, match="E2E_CONFIG_MISMATCH"):
        resolve_runtime_profile({"multitask": {"runtime": {"profile": None}}})
    assert not any(name == "ray" or name.startswith("verl.") for name in set(sys.modules) - before)


def test_deployment_checks_canonical_fixture_and_import(monkeypatch, tmp_path):
    result = launch_check.check_deployment(ROOT, ROOT / "D4_test.sh", ROOT / "multi_task_run.sh")
    assert result["source_root"] == str(ROOT.resolve())
    assert "trainer.py" in result["sha256"]
    old = tmp_path / "D4_test.sh"
    old.write_text("echo old-smoke\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="stale D4"):
        launch_check.check_deployment(ROOT, old, ROOT / "multi_task_run.sh")
    monkeypatch.setattr(importlib.util, "find_spec", lambda _: SimpleNamespace(origin=str(tmp_path / "wrong.py")))
    with pytest.raises(RuntimeError, match="plugin import mismatch"):
        launch_check.check_deployment(ROOT, ROOT / "D4_test.sh", ROOT / "multi_task_run.sh")


COLLISION = "MultiTaskCheckpointEngineWorker init_workers init_process_group TCPStore port: 37227 EADDRINUSE"


@pytest.mark.parametrize("log,retry", [
    (COLLISION, True),
    (COLLISION + "\nMULTITASK_TRAINING_COMPLETE", False),
    (COLLISION + "\nStarting FullyAsyncTrainer", False),
    (COLLISION + "\nray::Trainer.fit()", False),
    (COLLISION.replace("EADDRINUSE", "OOM"), False),
    ("vLLMHttpServer port: 37227 EADDRINUSE", False),
])
def test_only_pretraining_ce_collision_is_retryable(log, retry):
    assert startup_diagnostics.is_ce_startup_collision(log) is retry


def test_port_diagnostic_is_read_only_and_does_not_infer_owner(monkeypatch):
    calls = []
    monkeypatch.setattr(shutil, "which", lambda command: "/usr/bin/ss")
    monkeypatch.setattr(subprocess, "run", lambda args, **kwargs:
                        calls.append((args, kwargs)) or SimpleNamespace(returncode=0, stdout="", stderr=""))
    result = startup_diagnostics.diagnose(COLLISION)
    assert calls[0][0] == ["/usr/bin/ss", "-H", "-ltnp", "sport = :37227"]
    assert "earlier port owner" in result["note"]
    assert result["listeners_now"]["37227"]["stdout"] == ""


def test_ce_constructor_reports_selected_rendezvous_and_preserves_error(monkeypatch, capsys):
    path = ROOT / "src/multi_task_scheduler/checkpoint/checkpoint_engine_worker.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    original.body = [node for node in original.body if isinstance(node, ast.FunctionDef) and node.name == "__init__"]
    original.bases = [ast.Name(id="Parent", ctx=ast.Load())]
    error = RuntimeError("TCPStore port: 37227 EADDRINUSE")

    class Parent:
        def __init__(self, *args, **kwargs):
            raise error

    scope = {"Parent": Parent, "os": os, "json": json}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[original], type_ignores=[])), str(path), "exec"), scope)
    monkeypatch.setenv("MASTER_PORT", "37227")
    monkeypatch.setenv("RANK", "0")
    with pytest.raises(RuntimeError) as caught:
        scope[original.name](replica_rank=3)
    assert caught.value is error
    diagnostic = json.loads(capsys.readouterr().out.split("CE_RENDEZVOUS_CONFLICT ")[1])
    assert diagnostic["MASTER_PORT"] == "37227" and diagnostic["replica_rank"] == 3


def _bash():
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    path = str(git_bash) if sys.platform == "win32" and git_bash.is_file() else shutil.which("bash")
    if not path:
        pytest.skip("requires Bash for script wiring regression")
    return path


def _write(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8", newline="\n")


def test_batch_uses_checkout_fixture_even_when_outer_copy_is_old(tmp_path):
    outer = tmp_path / "server"
    plugin = outer / "verl/multi_task_verl"
    for name in ("D0_D4_batch_test.sh", "D0_D4_comprehensive_test.sh", "D4_test.sh"):
        _write(plugin / name, (ROOT / name).read_text(encoding="utf-8"))
    # These old neighboring copies must never be used. A copied *new* batch
    # must also select the checkout's comprehensive script rather than these.
    _write(outer / "D4_test.sh", "echo WRONG_OLD_D4; exit 99\n")
    _write(outer / "D0_D4_comprehensive_test.sh", "echo WRONG_OLD_COMPREHENSIVE; exit 99\n")
    _write(outer / "D0_D4_batch_test.sh", (ROOT / "D0_D4_batch_test.sh").read_text(encoding="utf-8"))
    # Explicit preflight double: this test proves shell path/argument wiring;
    # check_deployment and real verdict are tested separately above/elsewhere.
    _write(plugin / "src/multi_task_scheduler/testing/launch_check.py", "print('MOCK_PREFLIGHT')\n")
    _write(plugin / "src/multi_task_scheduler/testing/e2e_verdict.py", "raise SystemExit(88)\n")
    _write(outer / "multi_task_run.sh", """#!/usr/bin/env bash
printf 'SELECTED_SERVER_LAUNCHER required=%s scenario=%s steps=%s\\n' "$MULTITASK_E2E_REQUIRED" "$MULTITASK_E2E_SCENARIO" "$TOTAL_TRAINING_STEPS"
printf 'ARG=%s\\n' "$@"
exit 17
""")
    env = dict(os.environ, PYTHON_BIN=Path(sys.executable).as_posix(), PYTHONUTF8="1", D0_D4_BATCH_SCENARIOS="S1",
               VERL_REPO_DIR=outer.as_posix(), VERL_SOURCE_ROOT=(outer / "verl").as_posix(),
               VERL_MULTI_TASK_ROOT=plugin.as_posix())
    for key in ("MULTITASK_LAUNCH_SCRIPT", "D4_TOTAL_TRAINING_STEPS", "D0_D4_BATCH_LOG_DIR"):
        env.pop(key, None)
    proc = subprocess.run([_bash(), (outer / "D0_D4_batch_test.sh").as_posix()], env=env,
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "WRONG_OLD_" not in proc.stdout
    assert "SELECTED_SERVER_LAUNCHER required=1 scenario=basic steps=2" in proc.stdout, proc.stdout + proc.stderr
    assert "ARG=+multitask.e2e_test.enabled=true" in proc.stdout
    assert "ARG=+multitask.d4_runtime_test.enabled=false" in proc.stdout
    summaries = list((outer / "logs/d0_d4_batch").glob("*/summary.json"))
    assert len(summaries) == 1
    assert json.loads(summaries[0].read_text(encoding="utf-8"))["results"][0]["status"] == "FAIL"


@pytest.mark.parametrize("port_error,retries,attempts,status", [
    (True, "1", 2, "PASS"), (True, "0", 1, "FAIL"), (False, "1", 1, "FAIL"),
])
def test_s0_retries_only_the_complete_failed_ce_startup(tmp_path, port_error, retries, attempts, status):
    outer = tmp_path / "server"
    plugin = outer / "verl/multi_task_verl"
    for name in ("D0_D4_comprehensive_test.sh", "src/multi_task_scheduler/testing/startup_diagnostics.py"):
        _write(plugin / name, (ROOT / name).read_text(encoding="utf-8"))
    count_file = outer / "attempts.txt"
    first_error = COLLISION if port_error else "MultiTaskCheckpointEngineWorker init_workers OOM"
    _write(outer / "multi_task_run.sh", """#!/usr/bin/env bash
count=0
if [ -f "$FAKE_COUNT_FILE" ]; then
    read -r count < "$FAKE_COUNT_FILE"
fi
count=$((count + 1))
printf '%s\\n' "$count" > "$FAKE_COUNT_FILE"
if [ "$count" -eq 1 ]; then
    printf '%s\\n' "$FAKE_FIRST_ERROR"
    exit 1
fi
printf '%s\\n' 'MULTITASK_TRAINING_COMPLETE {"state": "COMPLETED", "completed": true}'
printf '%s\\n' 'CE_PARAMETER_VALIDATION {"state": "PARAMETERS_VALIDATED"}'
""")
    env = dict(os.environ, PYTHON_BIN=Path(sys.executable).as_posix(), PYTHONUTF8="1", D0_D4_SCENARIOS="S0",
               D0_D4_S0_PORT_RETRIES=retries, FAKE_COUNT_FILE=count_file.as_posix(), FAKE_FIRST_ERROR=first_error,
               VERL_REPO_DIR=outer.as_posix(), VERL_SOURCE_ROOT=(outer / "verl").as_posix(),
               VERL_MULTI_TASK_ROOT=plugin.as_posix())
    for key in ("MULTITASK_LAUNCH_SCRIPT", "D0_D4_LOG_DIR"):
        env.pop(key, None)
    proc = subprocess.run([_bash(), (plugin / "D0_D4_comprehensive_test.sh").as_posix()], env=env,
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
    assert proc.returncode == (0 if status == "PASS" else 1), proc.stdout + proc.stderr
    assert int(count_file.read_text()) == attempts
    summary = next((outer / "logs/d0_d4_comprehensive").glob("*/summary.json"))
    assert json.loads(summary.read_text(encoding="utf-8"))["results"][0]["status"] == status
    assert len(list(summary.parent.glob("S0_native_baseline_attempt_*.log"))) == attempts
