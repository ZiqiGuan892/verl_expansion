"""Mocked E2E orchestration tests; no real optimizer, Ray, vLLM, or NPU runs."""

import ast
import asyncio
import copy
import importlib.util
import json
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/multi_task_scheduler/testing/e2e_runtime.py"
MODULE_NAME = "multi_task_scheduler.testing.e2e_runtime"
spec = importlib.util.spec_from_file_location(MODULE_NAME, SOURCE)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def _reindex_claims(claims):
    """Execute the actual dependency-light claim reindexer with an NPU stub."""
    source = ROOT / "src/multi_task_scheduler/integration/verl/experimental_fully_async/llm_server_manager.py"
    parsed = ast.parse(source.read_text())
    cls = next(item for item in parsed.body if isinstance(item, ast.ClassDef))
    method = next(item for item in cls.body if isinstance(item, ast.FunctionDef)
                  and item.name == "_reindex_test_claims")
    method.decorator_list = []
    scope = {"get_device_name": lambda: "npu"}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), scope)
    return scope[method.name](claims)


def _prepare_fixture(worlds=(4,), world_size=4, pg_count=1):
    donors = [SimpleNamespace(replica_rank=i, world_size=world, allocation_kind="native")
              for i, world in enumerate(worlds)]
    claims = [{"node_id": "node", "accelerator_id": str(i + 4), "pg_id": f"pg-{i % pg_count}",
               "rank": i, "node_rank": 0, "local_rank": i, "claim_id": f"claim-{i}"}
              for i in range(world_size)]
    original = {"operation_id": "op", "lease_id": "lease", "borrower_replica_id": "borrowed",
                "world_size": world_size, "claims": claims,
                "parallelism": {"tensor_model_parallel_size": world_size}, "expires_at": 1}
    manager = SimpleNamespace(
        rollout_replicas=donors, server_addresses=["native"], server_handles=[object()],
        _build_d2_test_spec=AsyncMock(return_value=(original, False)),
        _local_native_donors=Mock(return_value=donors), _reindex_test_claims=_reindex_claims,
    )
    return SimpleNamespace(llm_server_manager=manager), manager, original


def test_prepare_split_creates_two_disjoint_tp2_contracts_from_all_four_claims():
    rollouter, manager, original = _prepare_fixture()
    unchanged = copy.deepcopy(original)
    result = asyncio.run(runtime.prepare(rollouter, "split"))
    manager._build_d2_test_spec.assert_awaited_once_with("basic")
    first, second = result["specs"]
    assert [part["world_size"] for part in result["specs"]] == [2, 2]
    assert {part["parallelism"]["tensor_model_parallel_size"] for part in result["specs"]} == {2}
    assert [[claim["rank"] for claim in part["claims"]] for part in result["specs"]] == [[0, 1], [0, 1]]
    assert [[claim["local_rank"] for claim in part["claims"]] for part in result["specs"]] == [[0, 1], [0, 1]]
    devices = [{claim["accelerator_id"] for claim in part["claims"]} for part in result["specs"]]
    assert devices[0].isdisjoint(devices[1]) and devices[0] | devices[1] == {"4", "5", "6", "7"}
    for key in ("lease_id", "operation_id", "borrower_replica_id"):
        assert first[key] != second[key]
    assert original == unchanged
    assert result["donor_ranks"] == [0] and result["donor_world_sizes"] == [4]
    assert rollouter._e2e_context["native"] == manager.rollout_replicas


@pytest.mark.parametrize("scenario", ["cross_pg", "merge_world_size"])
def test_prepare_cross_pg_requires_full_two_plus_two_merge(scenario):
    rollouter, manager, _ = _prepare_fixture(worlds=(2, 2), pg_count=2)
    result = asyncio.run(runtime.prepare(rollouter, scenario))
    manager._build_d2_test_spec.assert_awaited_once_with("merge_world_size")
    assert result["donor_world_sizes"] == [2, 2]
    assert [part["world_size"] for part in result["specs"]] == [4]
    assert len({claim["pg_id"] for claim in result["specs"][0]["claims"]}) == 2


@pytest.mark.parametrize("worlds,world_size,pg_count", [((4,), 4, 1), ((2, 2), 2, 2), ((2, 2), 4, 1)])
def test_prepare_rejects_partial_or_single_pg_merge(worlds, world_size, pg_count):
    rollouter, _, _ = _prepare_fixture(worlds, world_size, pg_count)
    with pytest.raises(ValueError, match="merge acceptance"):
        asyncio.run(runtime.prepare(rollouter, "cross_pg"))
    assert not hasattr(rollouter, "_e2e_context")


def _trainer_fixture():
    native = SimpleNamespace(replica_rank=0, allocation_kind="native", workers=[object(), object()])
    borrowed = SimpleNamespace(replica_rank=2, allocation_kind="borrowed", workers=[object()])
    excluded = SimpleNamespace(replica_rank=8, workers=[object()])
    manifest = {"complete": True, "global_steps": 3}
    validation = {"state": "PARAMETERS_VALIDATED", "source_state": "SOURCE_TO_RECEIVER_VALIDATED"}
    manager = SimpleNamespace(
        sync_state="IDLE", replicas=[native, borrowed, excluded],
        _effective_replicas_unlocked=Mock(return_value=[native, borrowed]),
        last_synced_versions={0: 3, 2: 3},
        _get_source_manifest=AsyncMock(return_value=manifest),
        validate_parameter_sync=AsyncMock(return_value=validation),
    )
    projection = AsyncMock()
    trainer = SimpleNamespace(
        checkpoint_manager=manager, current_param_version=3, _e2e_syncs=[],
        rollouter=SimpleNamespace(mark_replica_serving_version=SimpleNamespace(remote=projection)),
    )
    return trainer, manager, manifest, projection


def test_normal_sync_checks_effective_manifest_and_updates_borrowed_projection_only():
    trainer, manager, manifest, projection = _trainer_fixture()
    result = asyncio.run(runtime.record_normal_sync(trainer))
    manager.validate_parameter_sync.assert_awaited_once_with(
        manager._effective_replicas_unlocked.return_value, 3, source_manifest=manifest,
    )
    projection.assert_awaited_once_with(2, 3)
    assert result["replica_ranks"] == [0, 2]
    assert result["replica_worker_counts"] == {"0": 2, "2": 1}
    assert result["synchronized_versions"] == {"0": 3, "2": 3}
    assert result["origin"] == "optimizer_loop"
    assert trainer._e2e_syncs == [result]


@pytest.mark.parametrize("failure", ["unfinished", "stale", "manifest"])
def test_normal_sync_never_records_unconfirmed_effective_versions(failure):
    trainer, manager, _, projection = _trainer_fixture()
    if failure == "unfinished":
        manager.sync_state = "SYNCING"
    elif failure == "stale":
        manager.last_synced_versions[2] = 2
    else:
        manager.validate_parameter_sync.side_effect = RuntimeError("source manifest mismatch")
    with pytest.raises(RuntimeError):
        asyncio.run(runtime.record_normal_sync(trainer))
    assert trainer._e2e_syncs == []
    projection.assert_not_awaited()


class SyncRPC:
    """In-process RPC substitute; no actor scheduling or remote work occurs."""

    def __init__(self, callback):
        self.remote = callback


@pytest.mark.parametrize("scenario,failure", [
    ("basic", None), ("shared_bundle", None), ("concurrent_idempotent", None), ("pressure", None),
    ("basic", "bootstrap"), ("basic", "park"), ("basic", "bootstrap_cleanup"),
])
def test_training_fixture_orders_real_entry_points_with_mocked_rpc(monkeypatch, capsys, scenario, failure):
    events, validated = [], []
    shared = scenario == "shared_bundle"
    ranks = [1, 2] if shared else [1]
    active = [ranks[-1]] if shared else ranks
    world = 1 if shared else 4
    specs = [{"lease_id": f"lease-{rank}", "world_size": world, "claims": [{"node_id": "node"}]}
             for rank in ranks]
    version = {"current": 0}

    def rollout(action, payload):
        events.append(("rollout", action, copy.deepcopy(payload)))
        if action == "park" and failure == "park":
            raise RuntimeError("park failed")
        if action == "prepare":
            return {"specs": specs, "donor_ranks": [0]}
        if action == "owned":
            return {"replica_ranks": [] if failure == "park" else ranks}
        if action == "probe":
            return {"state": "GENERATED", "version": payload["version"],
                    "concurrency": payload.get("concurrency", 1), "routes_restored": True,
                    "inflight_before": 0, "inflight_after": 0,
                    "replicas": [{"replica_rank": rank, "server_id": f"server-{rank}", "requests": []}
                                 for rank in payload["replica_ranks"]]}
        if action == "topology":
            return {"training_replica_ranks": active, "replica_ranks": ranks}
        if action == "audit":
            return {"replicas": [{"replica_rank": rank, "completed": 2, "token_count": 2} for rank in active]}
        if action == "cleanup":
            if failure == "bootstrap_cleanup":
                raise RuntimeError("cleanup uncertain")
            return {"test_resources_released": True}
        return {}

    def train(action, payload):
        events.append(("trainer", action, copy.deepcopy(payload)))
        if action == "state":
            return {"version": version["current"], "normal_syncs": [{"version": 1}],
                    "training": {"completed": True, "current_param_version": 1}}
        if action in {"enable", "sync_current", "restore_donors"}:
            return {"version": version["current"], "parameter_validation": {"state": "PARAMETERS_VALIDATED"}}
        if action == "assert_removed":
            return {"ce_unregistered": True}
        return {}

    def member_call(name):
        return SyncRPC(lambda rank: events.append(("member", name, rank)) or {})

    trainer = SimpleNamespace(
        e2e_test_action=SyncRPC(train), suspend_donors_for_borrow=member_call("suspend"),
        unregister_replica=member_call("unregister"), register_replica=member_call("register"),
        bootstrap_replica=member_call("bootstrap"),
    )
    rollouter = SimpleNamespace(
        e2e_test_action=SyncRPC(rollout), commit_replica_ready=member_call("commit"),
        test_operation_snapshot=SyncRPC(lambda lease: {"server_id": "server-1", "worker_count": world,
                                                       "server_count": 1}),
    )
    lock = threading.Lock()
    calls = {"count": 0}

    def create(operation, contract):
        with lock:
            calls["count"] += 1
            count = calls["count"]
        rank = int(contract["lease_id"].split("-")[-1])
        events.append(("create", rank, count))
        if failure in {"bootstrap", "bootstrap_cleanup"}:
            raise RuntimeError("bootstrap failed after creating runtime")
        result = {"state": "LB_READY", "replica_rank": rank, "server_id": f"server-{rank}"}
        # The first submitted concurrent RPC may return the cached result;
        # only the other result contains the actual bootstrap receipt.
        if scenario != "concurrent_idempotent" or count == 2:
            result["bootstrap"] = {"version": 0}
        return result

    runner = SimpleNamespace(
        components={"config": {"multitask": {"e2e_test": {"enabled": True, "scenario": scenario}}},
                    "trainer": trainer, "rollouter": rollouter}, execute_replica_operation=create,
    )
    ray = ModuleType("ray")
    ray.get = lambda value: value
    monkeypatch.setitem(sys.modules, "ray", ray)
    verdict = ModuleType("multi_task_scheduler.testing.e2e_verdict")
    verdict.validate_result = lambda result, selected: validated.append((result, selected))
    monkeypatch.setitem(sys.modules, verdict.__name__, verdict)

    def native_fit():
        events.append(("native_fit",))
        version["current"] = 1  # Explicit substitute, not optimizer evidence.

    if failure:
        with pytest.raises(RuntimeError, match="park failed|bootstrap failed"):
            runtime.run_training_fixture(runner, native_fit)
        assert not validated and ("native_fit",) not in events
        owned = [] if failure == "park" else ranks
        assert ("rollout", "owned", {}) in events
        assert ("rollout", "cleanup", {"replica_ranks": owned}) in events
        if owned:
            assert ("trainer", "unregister", {"replica_rank": 1}) in events
        if failure == "bootstrap_cleanup":
            assert ("rollout", "wake_donors", {}) not in events
            assert ("rollout", "restore_routes", {}) not in events
        else:
            assert ("rollout", "wake_donors", {}) in events
            assert ("trainer", "restore_donors", {"replica_ranks": [0]}) in events
            assert events[-1] == ("rollout", "restore_routes", {})
        assert "D0_D4_E2E_RESULT " not in capsys.readouterr().out
        return
    runtime.run_training_fixture(runner, native_fit)
    result, selected = validated[0]
    assert selected == scenario
    fit_index = events.index(("native_fit",))
    audit_start = next(i for i, event in enumerate(events)
                       if event[:2] == ("rollout", "audit") and event[2].get("reset"))
    assert audit_start < fit_index
    assert all(event[2]["version"] == 0 for event in events[:fit_index] if event[:2] == ("rollout", "probe"))
    assert all(event[2]["version"] == 1 for event in events[fit_index:] if event[:2] == ("rollout", "probe"))
    cleanup_index = next(i for i, event in enumerate(events) if event[:2] == ("rollout", "cleanup"))
    restore_sync = next(i for i, event in enumerate(events) if event[:2] == ("trainer", "restore_donors"))
    restore_routes = next(i for i, event in enumerate(events) if event[:2] == ("rollout", "restore_routes"))
    assert fit_index < cleanup_index < restore_sync < restore_routes
    assert events[-1] == ("rollout", "probe", {"replica_ranks": [0], "version": 1})
    if shared:
        pause_a = events.index(("rollout", "pause", {"replica_rank": 1}))
        create_b = next(i for i, event in enumerate(events) if event[:2] == ("create", 2))
        pause_b = events.index(("rollout", "pause", {"replica_rank": 2}))
        wake_a = events.index(("rollout", "wake", {"replica_rank": 1}))
        assert events.index(("member", "unregister", 1)) < pause_a < create_b < fit_index < pause_b < wake_a
        assert wake_a < events.index(("member", "register", 1)) < events.index(("member", "bootstrap", 1))
        assert result["topology"]["activation_order"] == [1, 2, 1]
    if scenario == "concurrent_idempotent":
        assert calls["count"] == 2
        assert all(result["topology"]["idempotency"].values())
    if scenario == "pressure":
        assert result["generation_before"]["concurrency"] == result["generation_after"]["concurrency"] == 4
    marker = capsys.readouterr().out.split("D0_D4_E2E_RESULT ")[-1].strip()
    assert json.loads(marker) == result


def test_disabled_task_runner_delegates_native_loop_without_fixture(monkeypatch):
    path = ROOT / "src/multi_task_scheduler/integration/verl/experimental_fully_async/task_runner.py"
    parsed = ast.parse(path.read_text())
    cls = next(item for item in parsed.body if isinstance(item, ast.ClassDef))
    cls.bases = [ast.Name(id="Parent", ctx=ast.Load())]
    cls.decorator_list = []
    marker = object()

    class Parent:
        def _run_training_loop(self):
            return marker

    scope = {"Parent": Parent}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(path), "exec"), scope)
    module = ModuleType(MODULE_NAME)
    module.enabled = runtime.enabled
    module.run_training_fixture = Mock(side_effect=AssertionError("disabled fixture was called"))
    monkeypatch.setitem(sys.modules, MODULE_NAME, module)
    runner = scope[cls.name].__new__(scope[cls.name])
    runner.components = {"config": {}}
    assert runner._run_training_loop() is marker
    module.run_training_fixture.assert_not_called()


def test_owned_inventory_uses_all_fixture_leases_even_without_create_receipt(monkeypatch):
    cleanup = ModuleType("multi_task_scheduler.testing.e2e_cleanup")
    cleanup.capture_runtime = AsyncMock()
    cleanup.cleanup_runtime = AsyncMock()
    generation = ModuleType("multi_task_scheduler.testing.e2e_generation")
    generation.probe_replicas = AsyncMock()
    monkeypatch.setitem(sys.modules, cleanup.__name__, cleanup)
    monkeypatch.setitem(sys.modules, generation.__name__, generation)
    manager = SimpleNamespace(borrowed_operations={
        "first": {"replica_rank": 4, "state": "RUNTIME_READY"},
        "failed": {"replica_rank": 5, "state": "FAILED", "replica": None},
        "unrelated": {"replica_rank": 99},
    })
    rollouter = SimpleNamespace(llm_server_manager=manager, _e2e_context={
        "specs": [{"lease_id": "first"}, {"lease_id": "failed"}, {"lease_id": "not_created"}],
    })
    result = asyncio.run(runtime.dispatch_rollout(rollouter, "owned", {}))
    assert result["replica_ranks"] == [4, 5]


def test_cleanup_attempts_all_owned_runtimes_but_fails_if_any_is_uninspectable(monkeypatch):
    cleanup = ModuleType("multi_task_scheduler.testing.e2e_cleanup")
    cleanup.capture_runtime = AsyncMock(return_value={"captured": True})
    cleanup.cleanup_runtime = AsyncMock(return_value={"release_confirmed": True})
    generation = ModuleType("multi_task_scheduler.testing.e2e_generation")
    generation.probe_replicas = AsyncMock()
    monkeypatch.setitem(sys.modules, cleanup.__name__, cleanup)
    monkeypatch.setitem(sys.modules, generation.__name__, generation)
    monkeypatch.setitem(sys.modules, "ray", ModuleType("ray"))
    replica = SimpleNamespace(replica_rank=5, _server_address="borrowed-5")
    failed = {"replica_rank": 4, "replica": None}
    available = {"replica_rank": 5, "replica": replica}
    manager = SimpleNamespace(
        borrowed_operations={"failed": failed, "available": available}, rollout_replicas=[replica],
        ready_replica_ranks={5}, global_load_balancer=SimpleNamespace(
            get_total_inflight=SimpleNamespace(remote=AsyncMock(return_value=0)),
            remove_servers=SimpleNamespace(remote=AsyncMock()),
        ),
    )
    context = {"specs": [{"lease_id": "failed"}, {"lease_id": "available"}], "snapshots": {}, "cleaned": []}
    rollouter = SimpleNamespace(llm_server_manager=manager, _e2e_context=context)
    with pytest.raises(RuntimeError, match="not fully released"):
        asyncio.run(runtime.dispatch_rollout(rollouter, "cleanup", {"replica_ranks": []}))
    cleanup.capture_runtime.assert_awaited_once_with(replica)
    cleanup.cleanup_runtime.assert_awaited_once_with(replica, {"captured": True})
    assert available["state"] == "TEST_CLEANED"
    assert context["cleaned"] == [{"replica_rank": 5, "release_confirmed": True}]
