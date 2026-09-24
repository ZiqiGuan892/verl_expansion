"""D4 command-entry and READY publication tests without Ray or GPUs."""

import ast
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

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
