"""D3 CE membership and target-only bootstrap tests without Ray or GPUs."""

import ast
import asyncio
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest


SOURCE = Path(__file__).resolve().parents[2] / "src/multi_task_scheduler/checkpoint/checkpoint_engine_manager.py"


class _Factory:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _TargetGroup:
    world_size = 1

    def __init__(self, events):
        self.events = events

    def update_weights(self, **kwargs):
        self.events.append(("target_update", kwargs))
        return ["target-update"]

    def execute_checkpoint_engine(self, methods, **kwargs):
        self.events.append(("target_engine", methods))
        return ["target-engine"]


class _RayWorkerGroup:
    target = None

    @classmethod
    def from_detached(cls, **kwargs):
        return cls.target


class _Parent:
    def __init__(self, config, actor_wg, replicas):
        self.config = config
        self.backend = config.backend
        self.backend_cls = object()
        self.actor_wg = actor_wg
        self.replicas = list(replicas)

    async def update_weights(self, global_steps=None):
        self.config.events.append(("full_update", [r.replica_rank for r in self.replicas]))
        return {"native": True}


def _manager_class():
    parsed = ast.parse(SOURCE.read_text())
    node = next(item for item in parsed.body if isinstance(item, ast.ClassDef))
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )
    scope = {
        "TestParent": _Parent,
        "asyncio": asyncio,
        "copy": copy,
        "hashlib": hashlib,
        "json": json,
        "ray": SimpleNamespace(get=lambda refs: refs),
        "RayClassWithInitArgs": _Factory,
        "RayWorkerGroup": _RayWorkerGroup,
        "_worker_cls": object,
        "get_device_name": lambda: "cpu",
    }
    node.bases = [ast.Name(id="TestParent", ctx=ast.Load())]
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), scope)
    return scope["MultiTaskCheckpointEngineManager"]


def _replica(rank, worker):
    return SimpleNamespace(
        replica_rank=rank,
        workers=[worker],
        world_size=1,
        serving_version=None,
        abort_all_requests=AsyncMock(),
        release_kv_cache=AsyncMock(),
        resume_kv_cache=AsyncMock(),
        resume_generation=AsyncMock(),
    )


def _manager(events=None):
    events = [] if events is None else events
    actor_wg = SimpleNamespace(
        world_size=1,
        update_weights=Mock(side_effect=lambda **kwargs: events.append(("actor_update", kwargs)) or ["actor-update"]),
        execute_checkpoint_engine=Mock(
            side_effect=lambda methods, **kwargs: events.append(("actor_engine", methods)) or ["actor-engine"]
        ),
    )
    native = _replica(0, "native-worker")
    config = SimpleNamespace(backend="nccl", events=events)
    manager = _manager_class()(config=config, actor_wg=actor_wg, replicas=[native])
    manager.build_process_group = Mock(side_effect=lambda target: events.append(("build_topology", target.world_size)))
    return manager, events, native


def test_register_is_idempotent_and_pending_members_are_filtered_from_full_sync():
    manager, events, native = _manager()
    borrowed = _replica(3, "borrowed-worker")

    first = asyncio.run(manager.register_replica(borrowed))
    second = asyncio.run(manager.register_replica(borrowed))
    assert first["state"] == "REGISTERED"
    assert second["state"] == "ALREADY_REGISTERED"
    assert manager.pending_bootstrap == {3: None}

    asyncio.run(manager.update_weights(global_steps=2))
    assert ("full_update", [0]) in events
    assert manager.last_synced_versions == {0: 2}
    assert native.serving_version == 2

    with pytest.raises(ValueError, match="different workers"):
        asyncio.run(manager.register_replica(_replica(3, "other-worker")))


def test_target_only_bootstrap_finalizes_then_full_sync_includes_borrowed_replica():
    events = []
    manager, events, _ = _manager(events)
    borrowed = _replica(4, "borrowed-worker")
    target = _TargetGroup(events)
    target.world_size = 1
    _RayWorkerGroup.target = target

    asyncio.run(manager.register_replica(borrowed))
    result = asyncio.run(manager.bootstrap_replica(borrowed, snapshot_version=7))

    assert result == {
        "replica_rank": 4,
        "state": "WEIGHTS_READY",
        "version": 7,
        "communication": "target_only",
        "finalized": True,
    }
    assert manager.pending_bootstrap == {}
    assert manager.last_synced_versions[4] == 7
    assert borrowed.serving_version == 7
    assert events.index(("build_topology", 1)) < events.index(("actor_update", {"global_steps": 7, "mode": "nccl"}))
    assert any(name == "target_update" for name, _ in events)
    assert borrowed.abort_all_requests.await_count == 1
    assert borrowed.release_kv_cache.await_count == 1
    assert borrowed.resume_kv_cache.await_count == 1
    assert borrowed.resume_generation.await_count == 1

    asyncio.run(manager.update_weights(global_steps=8))
    assert ("full_update", [0, 4]) in events
    assert manager.last_synced_versions == {0: 8, 4: 8}


def test_suspended_donors_are_excluded_from_hccl_effective_set():
    events = []
    manager, events, _ = _manager(events)
    borrowed = _replica(4, "borrowed-worker")
    target = _TargetGroup(events)
    _RayWorkerGroup.target = target

    asyncio.run(manager.register_replica(borrowed))
    asyncio.run(manager.bootstrap_replica(borrowed, snapshot_version=7))
    result = asyncio.run(manager.suspend_replicas_for_sync([0]))

    assert result == {"state": "SUSPENDED", "replica_ranks": [0]}
    asyncio.run(manager.update_weights(global_steps=8))
    assert ("full_update", [4]) in events

    asyncio.run(manager.resume_replicas_for_sync([0]))
    asyncio.run(manager.update_weights(global_steps=9))
    assert ("full_update", [0, 4]) in events


def test_unregister_removes_pending_and_confirmed_version():
    manager, _, _ = _manager()
    borrowed = _replica(6, "borrowed-worker")
    asyncio.run(manager.register_replica(borrowed))
    manager.last_synced_versions[6] = 9

    result = asyncio.run(manager.unregister_replica(6))
    assert result["state"] == "UNREGISTERED"
    assert manager._find_replica_unlocked(6) is None
    assert 6 not in manager.pending_bootstrap
    assert 6 not in manager.last_synced_versions


def test_finalize_failure_does_not_confirm_bootstrap_or_allow_full_sync():
    manager, events, _ = _manager()
    borrowed = _replica(4, "borrowed-worker")
    target = _TargetGroup(events)
    target.execute_checkpoint_engine = Mock(side_effect=RuntimeError("finalize failed"))
    _RayWorkerGroup.target = target
    asyncio.run(manager.register_replica(borrowed))

    with pytest.raises(RuntimeError, match="finalize failed"):
        asyncio.run(manager.bootstrap_replica(borrowed, snapshot_version=7))

    assert manager.sync_state == "BLOCKED"
    assert 4 in manager.pending_bootstrap
    assert 4 not in manager.last_synced_versions
    assert borrowed.serving_version is None
    with pytest.raises(RuntimeError, match="BLOCKED"):
        asyncio.run(manager.update_weights(global_steps=8))


class _RemoteManifest:
    def __init__(self, manifest):
        self.manifest = manifest

    def remote(self):
        return self.manifest


class _ManifestWorker:
    def __init__(self, manifest):
        self.get_parameter_manifest = _RemoteManifest(manifest)


def _manifest(version=3, value="abc"):
    return {
        "complete": True,
        "global_steps": version,
        "wire_format": "named_tensors",
        "parameters": [
            {"name": "weight", "shape": [2], "dtype": "torch.float32", "numel": 2, "sha256": value}
        ],
        "parameter_count": 1,
        "total_numel": 2,
    }


def test_source_manifest_is_compared_parameter_by_parameter():
    manager, _, _ = _manager()
    source = _manifest()
    manager.actor_wg.execute_checkpoint_engine = Mock(return_value=[source])
    worker = _ManifestWorker(source)
    replica = _replica(4, worker)

    result = asyncio.run(manager.validate_parameter_sync([replica], expected_version=3, source_manifest=source))

    assert result["state"] == "PARAMETERS_VALIDATED"
    assert result["source_state"] == "SOURCE_TO_RECEIVER_VALIDATED"
    assert result["source_manifest_digest"] == result["manifest_digest"]


def test_source_manifest_mismatch_is_rejected():
    manager, _, _ = _manager()
    source = _manifest()
    received = _manifest(value="different")
    manager.actor_wg.execute_checkpoint_engine = Mock(return_value=[source])
    replica = _replica(4, _ManifestWorker(received))

    with pytest.raises(RuntimeError, match="differs from actor source manifest"):
        asyncio.run(manager.validate_parameter_sync([replica], expected_version=3, source_manifest=source))
