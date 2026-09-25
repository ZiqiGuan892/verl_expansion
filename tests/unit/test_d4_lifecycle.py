"""D4 command-entry and READY publication tests without Ray or GPUs."""

import ast
import asyncio
import copy
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _class_from_source(relative: str, name: str, parent, **scope):
    path = ROOT / relative
    parsed = ast.parse(path.read_text())
    node = next(item for item in parsed.body if isinstance(item, ast.ClassDef) and item.name == name)
    node.bases = [ast.Name(id="TestParent", ctx=ast.Load())]
    node.decorator_list = []
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )
    namespace = {"TestParent": parent, **scope}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


def test_commit_ready_is_idempotent_and_rejects_handle_replacement():
    class Parent:
        def __init__(self, servers, max_cache_size=10000, full_determinism=False):
            self._servers = dict(servers)
            self._inflight_requests = {key: 0 for key in servers}

    actor_a = SimpleNamespace(_actor_id="actor-a")
    actor_b = SimpleNamespace(_actor_id="actor-b")
    balancer_class = _class_from_source(
        "src/multi_task_scheduler/rollout/load_balancer.py",
        "MultiTaskGlobalRequestLoadBalancer",
        Parent,
        DEFAULT_ROUTING_CACHE_SIZE=10000,
    )
    balancer = balancer_class({}, group_scheduler=None)

    first = balancer.commit_ready({"server-a": actor_a})
    balancer._inflight_requests["server-a"] = 3
    second = balancer.commit_ready({"server-a": actor_a})

    assert first == {"state": "READY", "server_ids": ["server-a"], "added": ["server-a"]}
    assert second == {"state": "READY", "server_ids": ["server-a"], "added": []}
    assert balancer._inflight_requests["server-a"] == 3
    with pytest.raises(ValueError, match="already owned"):
        balancer.commit_ready({"server-a": actor_b})


def test_task_runner_create_operation_chains_runtime_ce_and_lb():
    events = []

    class Parent:
        def __init__(self):
            self.components = {}

    class RemoteCall:
        def __init__(self, callback):
            self.remote = callback

    runtime = {
        "operation_id": "op-1",
        "lease_id": "lease-1",
        "lease_ids": ["source-1"],
        "replica_rank": 7,
        "state": "RUNTIME_READY",
    }
    rollouter = SimpleNamespace(
        create_borrowed_replica=RemoteCall(lambda request: events.append("create") or runtime),
        commit_replica_ready=RemoteCall(lambda rank: events.append("ready") or {"state": "READY"}),
    )
    trainer = SimpleNamespace(
        register_replica=RemoteCall(lambda rank: events.append("register") or {"state": "REGISTERED"}),
        bootstrap_replica=RemoteCall(lambda rank: events.append("bootstrap") or {"state": "WEIGHTS_READY"}),
    )
    task_runner_class = _class_from_source(
        "src/multi_task_scheduler/integration/verl/experimental_fully_async/task_runner.py",
        "MultiTaskFullyAsyncTaskRunner",
        Parent,
        ray=SimpleNamespace(get=lambda value, **kwargs: value),
        threading=threading,
        logger=Mock(),
    )
    runner = task_runner_class()
    runner.components = {"rollouter": rollouter, "trainer": trainer}

    result = runner.execute_replica_operation("create", {"operation_id": "op-1", "lease_id": "lease-1"})

    assert events == ["create", "register", "bootstrap", "ready"]
    assert result["state"] == "LB_READY"
    assert result["replica_rank"] == 7


def test_task_runner_rejects_unsupported_lifecycle_operations_without_false_release():
    class Parent:
        def __init__(self):
            self.components = {}

    task_runner_class = _class_from_source(
        "src/multi_task_scheduler/integration/verl/experimental_fully_async/task_runner.py",
        "MultiTaskFullyAsyncTaskRunner",
        Parent,
        ray=SimpleNamespace(),
        threading=threading,
        logger=Mock(),
    )
    runner = task_runner_class()
    result = runner.execute_replica_operation("reclaim", {"lease_id": "lease-1"})
    assert result["state"] == "LIFECYCLE_NOT_IMPLEMENTED"
    assert result["released"] is False
    with pytest.raises(TypeError, match="mapping"):
        runner.execute_replica_operation("reclaim", ["not", "a", "mapping"])


@pytest.mark.parametrize("scenario,placement", [
    ("idempotent", "basic"), ("concurrent_idempotent", "basic"),
    ("basic", "basic"), ("split", "split"), ("fragmented", "fragmented"),
    ("cross_pg", "cross_pg"), ("merge_world_size", "merge_world_size"),
])
def test_d4_retry_scenarios_use_basic_placement_and_leave_other_scenarios_unchanged(scenario, placement):
    rollouter_class = _class_from_source(
        "src/multi_task_scheduler/integration/verl/experimental_fully_async/rollouter.py",
        "MultiTaskFullyAsyncRollouter", object,
    )
    rollouter = rollouter_class.__new__(rollouter_class)
    spec = {"lease_id": "test-lease"}
    sleeping = {"replica_ranks": [0]}
    manager = SimpleNamespace(
        _build_d2_test_spec=AsyncMock(return_value=(spec, False)),
        sleep_d4_test_donors=AsyncMock(return_value=sleeping),
    )
    rollouter.llm_server_manager = manager
    result = asyncio.run(rollouter.prepare_d4_runtime_smoke(scenario))
    manager._build_d2_test_spec.assert_awaited_once_with(placement)
    manager.sleep_d4_test_donors.assert_awaited_once_with(spec)
    assert result == {"spec": spec, "expected_failure": False, "sleeping": sleeping}


@pytest.mark.parametrize("scenario", ["idempotent", "concurrent_idempotent"])
@pytest.mark.parametrize("worker_count,server_count,valid", [(4, 1, True), (1, 1, False), (8, 2, False)])
def test_d4_idempotency_smoke_checks_replica_topology_not_single_worker(
    scenario, worker_count, server_count, valid, capsys
):
    """Execute the whole smoke hook with synchronous RPC substitutes, including thread retries."""
    runner_class = _class_from_source(
        "src/multi_task_scheduler/integration/verl/experimental_fully_async/task_runner.py",
        "MultiTaskFullyAsyncTaskRunner", object,
        ray=SimpleNamespace(get=lambda value, **kwargs: value), threading=threading,
        copy=copy, json=json, ThreadPoolExecutor=ThreadPoolExecutor, logger=Mock(),
    )
    runner = runner_class()
    spec = {"lease_id": "lease", "world_size": 4, "claims": [{"node_id": "node"} for _ in range(4)]}
    runtime = {"state": "RUNTIME_READY", "replica_rank": 1, "lease_id": "lease"}
    ready = {"state": "LB_READY", "replica_rank": 1, "server_id": "endpoint"}
    snapshot = {"worker_count": worker_count, "server_count": server_count, "server_id": "endpoint"}

    def rpc(value):
        return SimpleNamespace(remote=Mock(return_value=value))

    rollouter = SimpleNamespace(
        prepare_d4_runtime_smoke=rpc({"spec": spec, "expected_failure": False,
                                     "sleeping": {"replica_ranks": [0]}}),
        create_borrowed_replica=SimpleNamespace(remote=Mock(side_effect=[runtime, ready])),
        commit_replica_ready=rpc(ready), test_operation_snapshot=rpc(snapshot),
        probe_replica_ready=rpc({"registered": True}), cleanup_d4_runtime=rpc({"state": "CLEANED"}),
    )
    trainer = SimpleNamespace(**{name: rpc({}) for name in (
        "register_replica", "bootstrap_replica", "unregister_replica",
        "suspend_donors_for_borrow", "resume_donors_after_borrow",
    )})
    runner.components = {"rollouter": rollouter, "trainer": trainer}
    config = {"multitask": {"d4_runtime_test": {"enabled": True, "scenario": scenario}}}
    if valid:
        runner._maybe_run_d4_runtime_smoke(config)
        output = capsys.readouterr().out
        marker = "D4_IDEMPOTENCY_RESULT" if scenario == "idempotent" else "D4_CONCURRENCY_RESULT"
        receipt = json.loads(next(line.split(" ", 1)[1] for line in output.splitlines() if line.startswith(marker)))
        assert receipt["same_rank"] and receipt["same_server"]
        assert receipt["worker_count"] == receipt["expected_worker_count"] == 4
        assert receipt["server_count"] == receipt["expected_server_count"] == 1
    else:
        with pytest.raises(RuntimeError, match="idempotency"):
            runner._maybe_run_d4_runtime_smoke(config)
    assert rollouter.create_borrowed_replica.remote.call_count == 2
    trainer.register_replica.remote.assert_called_once_with(1)
    trainer.bootstrap_replica.remote.assert_called_once_with(1)
    rollouter.commit_replica_ready.remote.assert_called_once_with(1)
    trainer.unregister_replica.remote.assert_called_once_with(1)
    rollouter.cleanup_d4_runtime.remote.assert_called_once_with(1)
    trainer.resume_donors_after_borrow.remote.assert_called_once_with([0])
