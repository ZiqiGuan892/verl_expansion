"""D1 contract tests using isolated class bodies.

These tests intentionally do not import verl, Ray, vLLM, or NPU backends.  The
manager and replica classes are executed with a small substituted native
parent so the tests cover the real D1 method bodies without claiming runtime
creation success.
"""

import ast
import asyncio
import copy
import math
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


SOURCE = Path(__file__).resolve().parents[2] / "src/multi_task_scheduler"


def _isolated_class(relative: str, name: str, parent: type, **globals_for_test):
    path = SOURCE / relative
    parsed = ast.parse(path.read_text())
    node = next(item for item in parsed.body if isinstance(item, ast.ClassDef) and item.name == name)
    node.bases = [ast.Name(id="TestParent", ctx=ast.Load())]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )
    scope = {
        "TestParent": parent,
        "asyncio": asyncio,
        "copy": copy,
        "math": math,
        "time": time,
        **globals_for_test,
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return scope[name]


class _ManagerParent:
    def __init__(self, config, worker_group=None, rollout_resource_pool=None):
        self.config = config
        self.rollout_config = config.actor_rollout_ref.rollout
        self.worker_group = worker_group
        self.rollout_resource_pool = rollout_resource_pool
        self.rollout_replicas = []
        self.hybrid_replicas = {}
        self.alive_replicas = {}


def _manager_class():
    return _isolated_class(
        "integration/verl/experimental_fully_async/llm_server_manager.py",
        "MultiTaskLLMServerManager",
        _ManagerParent,
        ray=SimpleNamespace(remote=Mock()),
        MultiTaskvLLMReplica=object,
        MultiTaskGlobalRequestLoadBalancer=object,
    )


def _config(max_colocate_count=4):
    return SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(max_colocate_count=max_colocate_count))
    )


def _claim(index, *, node_id=None, node_rank=None, local_rank=None, claim_id=None, lease_id=None):
    return {
        "claim_id": claim_id or f"claim-{index}",
        "lease_id": lease_id or f"source-{index % 2}",
        "donor_task_id": f"donor-task-{index % 2}",
        "donor_replica_rank": index % 2,
        "pg_id": f"pg-{index % 2}",
        "bundle_index": index % 2,
        "node_id": node_id or f"node-{index // 2}",
        "gpu_uuid": f"gpu-{index}",
        "local_gpu_index": index % 2,
        "node_rank": index // 2 if node_rank is None else node_rank,
        "local_rank": index % 2 if local_rank is None else local_rank,
        "gpu_fraction": 0.25,
        "cpu_request": 1.0,
    }


def _spec(**overrides):
    value = {
        "operation_id": "operation-1",
        "lease_id": "borrower-lease-1",
        "lease_ids": ["source-0", "source-1"],
        "borrower_task_id": "borrower-task-1",
        "borrower_replica_id": "borrower-replica-opaque",
        "replica_rank": None,
        "claims": [_claim(i) for i in range(4)],
        "world_size": 4,
        "max_colocate_count": 4,
        "expires_at": time.time() + 600,
        "placement_epoch": 3,
    }
    value.update(overrides)
    return value


def _manager():
    manager_class = _manager_class()
    return manager_class(_config())


def test_validate_create_spec_normalizes_claims_without_mutating_input():
    manager = _manager()
    original = _spec()
    normalized = manager._validate_create_spec(original)

    assert original["claims"][0].get("rank") is None
    assert normalized["claims"] == sorted(normalized["claims"], key=lambda claim: claim["rank"])
    assert [claim["rank"] for claim in normalized["claims"]] == [0, 1, 2, 3]
    assert normalized["source_lease_ids"] == ["source-0", "source-1"]
    assert normalized["world_size"] == len(normalized["claims"]) == 4


@pytest.mark.parametrize(
    "mutator,pattern",
    [
        (lambda spec: spec.update(world_size=3), "world_size"),
        (lambda spec: spec["claims"].__setitem__(1, copy.deepcopy(spec["claims"][0])), "duplicate claim_id"),
        (lambda spec: spec.update(expires_at=time.time() - 1), "expired"),
        (lambda spec: spec["claims"][0].update(gpu_fraction=1.1), "gpu_fraction"),
        (lambda spec: spec["claims"][3].update(node_id="node-extra", node_rank=2, local_rank=0), "uniform"),
    ],
)
def test_validate_create_spec_rejects_invalid_contract(mutator, pattern):
    manager = _manager()
    value = _spec()
    mutator(value)
    with pytest.raises(ValueError, match=pattern):
        manager._validate_create_spec(value)


def test_create_contract_is_idempotent_and_does_not_start_runtime():
    manager = _manager()
    request = _spec()
    first = asyncio.run(manager.create_borrowed_replica(request))
    retry = copy.deepcopy(request)
    retry["operation_id"] = "operation-retry"
    second = asyncio.run(manager.create_borrowed_replica(retry))

    assert first == second
    assert first["state"] == "FAILED"
    assert first["released"] is False
    assert first["error"]["code"] == "RUNTIME_CREATION_NOT_IMPLEMENTED"
    assert first["replica_rank"] == 0
    assert len(manager.borrowed_operations) == 1
    assert manager.borrowed_operations["borrower-lease-1"]["replica"] is None

    different = _spec(
        operation_id="operation-2",
        lease_id="borrower-lease-2",
        borrower_replica_id="borrower-replica-2",
    )
    third = asyncio.run(manager.create_borrowed_replica(different))
    assert third["replica_rank"] == 1
    conflicting_operation = _spec(
        operation_id="operation-1",
        lease_id="borrower-lease-3",
        borrower_replica_id="borrower-replica-3",
    )
    with pytest.raises(ValueError, match="operation_id"):
        asyncio.run(manager.create_borrowed_replica(conflicting_operation))


def test_concurrent_duplicate_requests_allocate_one_rank_and_reclaim_is_explicitly_unimplemented():
    manager = _manager()
    base = _spec()
    requests = []
    for i in range(8):
        request = copy.deepcopy(base)
        request["operation_id"] = f"operation-{i}"
        requests.append(request)

    async def run():
        return await asyncio.gather(*(manager.create_borrowed_replica(item) for item in requests))

    receipts = asyncio.run(run())
    assert {receipt["replica_rank"] for receipt in receipts} == {0}
    assert len(manager.borrowed_operations) == 1

    reclaim = asyncio.run(manager.reclaim_replica("borrower-lease-1"))
    assert reclaim["released"] is False
    assert reclaim["error"]["code"] == "LIFECYCLE_NOT_IMPLEMENTED"


def test_rank_is_monotonic_and_retire_requires_the_owner():
    manager = _manager()
    first = asyncio.run(manager.create_borrowed_replica(_spec()))
    asyncio.run(manager.retire_replica_rank(first["replica_rank"], "borrower-task-1"))
    assert first["replica_rank"] in manager.retired_replica_ranks

    second = asyncio.run(
        manager.create_borrowed_replica(
            _spec(
                operation_id="operation-2",
                lease_id="borrower-lease-2",
                borrower_replica_id="borrower-replica-2",
            )
        )
    )
    assert second["replica_rank"] == 1
    with pytest.raises(KeyError):
        asyncio.run(manager.retire_replica_rank(second["replica_rank"], "wrong-owner"))


class _ReplicaParent:
    def __init__(self, *args, **kwargs):
        self.replica_rank = kwargs.get("replica_rank", args[0] if args else 0)


def _replica_class():
    return _isolated_class(
        "rollout/replica.py",
        "MultiTaskvLLMReplica",
        _ReplicaParent,
        ray=SimpleNamespace(remote=Mock(return_value=object())),
        RayClassWithInitArgs=object,
        MultiTaskCheckpointEngineWorker=object,
        MultiTaskvLLMHttpServer=object,
    )


def test_replica_contract_keeps_native_and_borrowed_metadata_separate():
    replica_class = _replica_class()
    native = replica_class(replica_rank=0)
    borrowed = replica_class(
        replica_rank=3,
        allocation_kind="borrowed",
        lease_id="borrower-lease-1",
        source_lease_ids=["source-0"],
        owns_resource_pool=False,
        claims=[{"claim_id": "claim-0"}],
    )

    assert native.allocation_kind == "native"
    assert native.lease_id is None
    assert native.owns_resource_pool is True
    assert borrowed.allocation_kind == "borrowed"
    assert borrowed.lease_id == "borrower-lease-1"
    assert borrowed.owns_resource_pool is False
    assert borrowed.claims == [{"claim_id": "claim-0"}]

    with pytest.raises(NotImplementedError, match="deferred to D2"):
        asyncio.run(borrowed.init_from_lease({}))
    assert borrowed.runtime_state == "FAILED"
    mismatch = asyncio.run(borrowed.reclaim("other-lease"))
    assert mismatch["error"]["code"] == "LEASE_MISMATCH"
    assert asyncio.run(borrowed.destroy())["released"] is False
