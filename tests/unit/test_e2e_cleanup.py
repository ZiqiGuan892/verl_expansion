"""Mocked teardown contracts; these tests do not prove Ray/GPU process release."""

import asyncio
import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest


SOURCE = Path(__file__).resolve().parents[2] / "src/multi_task_scheduler/testing/e2e_cleanup.py"
SPEC = importlib.util.spec_from_file_location("e2e_cleanup_under_test", SOURCE)
cleanup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cleanup)


def fixture_runtime():
    servers = [SimpleNamespace(_actor_id="http")]
    workers = [SimpleNamespace(_actor_id="ce")]
    replica = SimpleNamespace(
        allocation_kind="borrowed", owns_resource_pool=False, replica_rank=7,
        servers=servers, workers=workers,
        _cleanup_runtime=AsyncMock(return_value={"kill_requested": ["http", "ce"], "errors": []}),
    )
    actors = [
        {"actor_id": "http", "node_id": "node-1", "role": "server", "pid": 10, "create_time": 1.0,
         "children": [{"pid": 11, "create_time": 2.0}], "child_tree_complete": True,
         "endpoint": {"host": "127.0.0.1", "port": 8080}},
        {"actor_id": "ce", "node_id": "node-1", "role": "worker", "pid": 20, "create_time": 3.0,
         "children": [], "child_tree_complete": False, "endpoint": None},
    ]
    for handle, record in zip(servers + workers, actors, strict=True):
        handle.__ray_call__ = SimpleNamespace(remote=AsyncMock(return_value=copy.deepcopy(record)))
    return replica, {"replica_rank": 7, "actors": actors}


def released_node():
    exited = {"gone": True, "exited": True, "reaped": True}
    return {"node_id": "node-1", "actors": [dict(exited), dict(exited)],
            "children": [dict(exited)], "endpoints": [{"closed": True}]}


def test_cleanup_requires_all_three_independent_release_checks(monkeypatch):
    replica, snapshot = fixture_runtime()
    monkeypatch.setattr(cleanup, "_actor_dead", AsyncMock(return_value={"dead": True}))
    observe = AsyncMock(return_value=released_node())
    monkeypatch.setattr(cleanup, "_observe_node", observe)
    result = asyncio.run(cleanup.cleanup_runtime(replica, snapshot, timeout_s=0.1))
    assert result["state"] == "TEST_RUNTIME_CLEANED"
    assert result["release_confirmed"] is True
    assert result["released"] is False
    assert result["actors_dead"] and result["child_processes_dead"] and result["endpoint_closed"]
    assert result["process_table_reaped"] is True and result["unreaped_processes"] == []
    replica._cleanup_runtime.assert_awaited_once()
    assert observe.await_args.args[0] == snapshot["actors"]
    assert 0 < observe.await_args.args[2] <= 0.1 + 1e-9


def test_default_cleanup_allows_cold_node_observer_startup(monkeypatch):
    replica, snapshot = fixture_runtime()
    monkeypatch.setattr(cleanup, "_actor_dead", AsyncMock(return_value={"dead": True}))
    observe = AsyncMock(return_value=released_node())
    monkeypatch.setattr(cleanup, "_observe_node", observe)
    result = asyncio.run(cleanup.cleanup_runtime(replica, snapshot))
    assert result["release_confirmed"] is True
    assert observe.await_args.args[2] == 30.0


@pytest.mark.parametrize("missing", ["actor_rpc", "actor_process", "child", "endpoint"])
def test_any_missing_release_evidence_times_out(monkeypatch, missing):
    replica, snapshot = fixture_runtime()
    node = released_node()
    if missing == "actor_process":
        node["actors"][0].update(gone=False, exited=False, reaped=False)
    elif missing == "child":
        node["children"][0].update(gone=False, exited=False, reaped=False)
    elif missing == "endpoint":
        node["endpoints"][0]["closed"] = False
    monkeypatch.setattr(cleanup, "_actor_dead", AsyncMock(return_value={"dead": missing != "actor_rpc"}))
    monkeypatch.setattr(cleanup, "_observe_node", AsyncMock(return_value=node))
    with pytest.raises(cleanup.RuntimeCleanupError) as caught:
        asyncio.run(cleanup.cleanup_runtime(replica, snapshot, timeout_s=0.01))
    assert caught.value.diagnostics["release_confirmed"] is False
    assert caught.value.diagnostics["errors"]


def test_node_inspection_error_cannot_become_success(monkeypatch):
    replica, snapshot = fixture_runtime()
    monkeypatch.setattr(cleanup, "_actor_dead", AsyncMock(return_value={"dead": True}))
    monkeypatch.setattr(cleanup, "_observe_node", AsyncMock(side_effect=PermissionError("process access denied")))
    with pytest.raises(cleanup.RuntimeCleanupError, match="process access denied"):
        asyncio.run(cleanup.cleanup_runtime(replica, snapshot))


def test_kill_error_is_preserved_even_if_processes_later_disappear(monkeypatch):
    replica, snapshot = fixture_runtime()
    replica._cleanup_runtime.return_value["errors"] = ["kill failed"]
    monkeypatch.setattr(cleanup, "_actor_dead", AsyncMock(return_value={"dead": True}))
    monkeypatch.setattr(cleanup, "_observe_node", AsyncMock(return_value=released_node()))
    with pytest.raises(cleanup.RuntimeCleanupError, match="kill failed"):
        asyncio.run(cleanup.cleanup_runtime(replica, snapshot))


@pytest.mark.parametrize("invalid", ["native", "foreign_actor", "missing_child", "actor_as_child"])
def test_invalid_ownership_cannot_issue_kill(invalid):
    replica, snapshot = fixture_runtime()
    if invalid == "native":
        replica.allocation_kind = "native"
    elif invalid == "foreign_actor":
        snapshot["actors"][0]["actor_id"] = "foreign"
    elif invalid == "missing_child":
        snapshot["actors"][0]["children"] = []
    else:
        snapshot["actors"][0]["children"][0]["pid"] = 20
    with pytest.raises(ValueError):
        asyncio.run(cleanup.cleanup_runtime(replica, snapshot))
    replica._cleanup_runtime.assert_not_called()


def test_reused_pid_is_never_signaled(monkeypatch):
    process = SimpleNamespace(create_time=lambda: 9.0, terminate=Mock(), kill=Mock())
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(
        Process=lambda pid: process, NoSuchProcess=ProcessLookupError,
    ))
    result = cleanup._inspect_process({"pid": 12, "create_time": 1.0}, terminate="kill")
    assert result["state"] == "PID_REUSED" and result["gone"]
    process.terminate.assert_not_called()
    process.kill.assert_not_called()


def test_matching_child_identity_can_be_signaled_but_is_not_yet_gone(monkeypatch):
    process = SimpleNamespace(pid=12, create_time=lambda: 1.0, status=lambda: "running", terminate=Mock(), kill=Mock())
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(
        Process=lambda pid: process, NoSuchProcess=ProcessLookupError,
    ))
    result = cleanup._inspect_process({"pid": 12, "create_time": 1.0}, terminate="terminate")
    process.terminate.assert_called_once()
    assert result["gone"] is False and result["exited"] is False


@pytest.mark.parametrize("status", ["zombie", "dead"])
def test_linux_terminal_process_has_exited_without_being_reaped_or_signaled(monkeypatch, status):
    process = SimpleNamespace(
        pid=12, create_time=lambda: 1.0, status=lambda: status,
        threads=lambda: [SimpleNamespace(id=12)], terminate=Mock(), kill=Mock(),
    )
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(
        Process=lambda pid: process, NoSuchProcess=ProcessLookupError,
    ))
    result = cleanup._inspect_process({"pid": 12, "create_time": 1.0}, terminate="kill")
    assert result["exited"] is True
    assert result["gone"] is False and result["reaped"] is False
    process.terminate.assert_not_called()
    process.kill.assert_not_called()


def test_zombie_leader_with_executing_thread_is_not_whole_process_exit(monkeypatch):
    process = SimpleNamespace(
        pid=12, create_time=lambda: 1.0, status=lambda: "zombie",
        threads=lambda: [SimpleNamespace(id=12), SimpleNamespace(id=13)], terminate=Mock(), kill=Mock(),
    )
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(
        Process=lambda pid: process if pid == 12 else SimpleNamespace(status=lambda: "running"),
        NoSuchProcess=ProcessLookupError,
    ))
    result = cleanup._inspect_process({"pid": 12, "create_time": 1.0})
    assert result["exited"] is False and result["reaped"] is False


def test_zombie_exception_is_not_misclassified_as_missing_process(monkeypatch):
    class ZombieProcess(ProcessLookupError):
        pass

    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(
        Process=Mock(side_effect=ZombieProcess("cannot read identity")),
        NoSuchProcess=ProcessLookupError, ZombieProcess=ZombieProcess,
    ))
    with pytest.raises(RuntimeError, match="could not verify terminal process identity"):
        cleanup._inspect_process({"pid": 12, "create_time": 1.0})


def test_exited_unreaped_child_does_not_block_execution_resource_release(monkeypatch):
    replica, snapshot = fixture_runtime()
    node = released_node()
    node["children"][0].update(gone=False, exited=True, reaped=False, state="zombie", pid=11, create_time=2.0)
    monkeypatch.setattr(cleanup, "_actor_dead", AsyncMock(return_value={"dead": True}))
    monkeypatch.setattr(cleanup, "_observe_node", AsyncMock(return_value=node))
    result = asyncio.run(cleanup.cleanup_runtime(replica, snapshot, timeout_s=0.1))
    assert result["release_confirmed"] is True and result["child_processes_dead"] is True
    assert result["process_table_reaped"] is False
    assert result["unreaped_processes"][0]["pid"] == 11


def test_temporary_ray_actor_unavailability_does_not_confirm_death(monkeypatch):
    class ActorError(Exception):
        pass

    class UnavailableError(ActorError):
        pass

    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(exceptions=SimpleNamespace(
        RayActorError=ActorError, ActorUnavailableError=UnavailableError,
    )))
    actor = SimpleNamespace(__ray_call__=SimpleNamespace(remote=AsyncMock(side_effect=UnavailableError())))
    assert asyncio.run(cleanup._actor_dead(actor, 0.1))["dead"] is False
    actor.__ray_call__.remote.side_effect = ActorError()
    assert asyncio.run(cleanup._actor_dead(actor, 0.1))["dead"] is True


def test_capture_reads_each_actual_actor_and_preserves_process_tree():
    replica, snapshot = fixture_runtime()
    for handle, record in zip(replica.servers + replica.workers, snapshot["actors"], strict=True):
        handle.__ray_call__ = SimpleNamespace(remote=AsyncMock(return_value=copy.deepcopy(record)))
    captured = asyncio.run(cleanup.capture_runtime(replica))
    assert captured["actors"] == snapshot["actors"]
    for handle, role in [(replica.servers[0], "server"), (replica.workers[0], "worker")]:
        handle.__ray_call__.remote.assert_awaited_once_with(cleanup._capture_actor, role)


def test_module_import_requires_no_ray_or_psutil(monkeypatch):
    monkeypatch.setitem(sys.modules, "ray", None)
    monkeypatch.setitem(sys.modules, "psutil", None)
    module = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(module)
    assert callable(module.capture_runtime) and callable(module.cleanup_runtime)


def test_node_observer_signals_only_children_and_requires_connection_refusal(monkeypatch):
    import errno

    _, snapshot = fixture_runtime()
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(
        get_runtime_context=lambda: SimpleNamespace(get_node_id=lambda: "node-1"),
    ))
    inspect = Mock(return_value={"gone": True})
    monkeypatch.setattr(cleanup, "_inspect_process", inspect)
    connection = Mock(side_effect=ConnectionRefusedError(errno.ECONNREFUSED, "closed"))
    monkeypatch.setitem(sys.modules, "socket", SimpleNamespace(create_connection=connection))
    observed = cleanup._inspect_node(snapshot["actors"], terminate="kill")
    assert observed["endpoints"][0]["closed"] is True
    assert inspect.call_args_list[0].kwargs == {}
    assert inspect.call_args_list[1].args == ({"pid": 11, "create_time": 2.0},)
    assert inspect.call_args_list[1].kwargs == {"terminate": "kill"}
    assert inspect.call_args_list[2].kwargs == {}
    connection.side_effect = TimeoutError("connection timed out")
    with pytest.raises(TimeoutError):
        cleanup._inspect_node(snapshot["actors"])


def test_wrong_node_cannot_inspect_or_signal_local_pids(monkeypatch):
    _, snapshot = fixture_runtime()
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(
        get_runtime_context=lambda: SimpleNamespace(get_node_id=lambda: "different-node"),
    ))
    inspect = Mock()
    monkeypatch.setattr(cleanup, "_inspect_process", inspect)
    with pytest.raises(RuntimeError, match="different node"):
        cleanup._inspect_node(snapshot["actors"], terminate="kill")
    inspect.assert_not_called()


def test_observer_uses_zero_cpus_and_hard_node_affinity(monkeypatch):
    _, snapshot = fixture_runtime()
    remote_call = SimpleNamespace(remote=AsyncMock(return_value=released_node()))
    options = Mock(return_value=remote_call)
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(remote=lambda fn: SimpleNamespace(options=options)))
    affinity = Mock(return_value="affinity-to-node-1")
    monkeypatch.setitem(sys.modules, "ray.util.scheduling_strategies", SimpleNamespace(
        NodeAffinitySchedulingStrategy=affinity,
    ))
    result = asyncio.run(cleanup._observe_node(snapshot["actors"], None, 0.1))
    assert result == released_node()
    affinity.assert_called_once_with(node_id="node-1", soft=False)
    options.assert_called_once_with(num_cpus=0, max_retries=0, scheduling_strategy="affinity-to-node-1")


def test_capture_fails_when_process_tree_changes(monkeypatch):
    process = SimpleNamespace(pid=10, create_time=lambda: 1.0, children=Mock(side_effect=[
        [SimpleNamespace(pid=11, create_time=lambda: 2.0)],
        [SimpleNamespace(pid=12, create_time=lambda: 3.0)],
    ]))
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(Process=lambda pid: process))
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(
        get_runtime_context=lambda: SimpleNamespace(get_actor_id=lambda: "http", get_node_id=lambda: "node-1"),
    ))
    with pytest.raises(RuntimeError, match="process tree changed"):
        cleanup._capture_actor(SimpleNamespace(node_rank=0), "server")


def test_cleanup_refresh_unions_new_and_previously_captured_children(monkeypatch):
    replica, snapshot = fixture_runtime()
    refreshed = copy.deepcopy(snapshot)
    refreshed["captured_at"] = 5.0
    refreshed["actors"][0]["children"] = [{"pid": 30, "create_time": 4.0}]
    capture = AsyncMock(return_value=refreshed)
    observe = AsyncMock(return_value=released_node())
    monkeypatch.setattr(cleanup, "capture_runtime", capture)
    monkeypatch.setattr(cleanup, "_actor_dead", AsyncMock(return_value={"dead": True}))
    monkeypatch.setattr(cleanup, "_observe_node", observe)
    result = asyncio.run(cleanup.cleanup_runtime(replica, snapshot, timeout_s=0.1))
    assert result["capture_refreshed"] is True
    assert observe.await_args.args[0][0]["children"] == [
        {"pid": 11, "create_time": 2.0}, {"pid": 30, "create_time": 4.0},
    ]
    assert snapshot["actors"][0]["children"] == [{"pid": 11, "create_time": 2.0}]


def test_failed_refresh_still_tears_down_known_actors_but_cannot_pass(monkeypatch):
    replica, snapshot = fixture_runtime()
    monkeypatch.setattr(cleanup, "capture_runtime", AsyncMock(side_effect=RuntimeError("HTTP actor died")))
    monkeypatch.setattr(cleanup, "_actor_dead", AsyncMock(return_value={"dead": True}))
    monkeypatch.setattr(cleanup, "_observe_node", AsyncMock(return_value=released_node()))
    with pytest.raises(cleanup.RuntimeCleanupError, match="HTTP actor died") as caught:
        asyncio.run(cleanup.cleanup_runtime(replica, snapshot, timeout_s=0.1))
    assert caught.value.diagnostics["capture_refreshed"] is False
    assert caught.value.diagnostics["release_confirmed"] is False
    replica._cleanup_runtime.assert_awaited_once()


@pytest.mark.parametrize("changed", ["node_id", "pid", "create_time", "endpoint"])
def test_changed_actor_identity_or_endpoint_invalidates_refresh(monkeypatch, changed):
    replica, snapshot = fixture_runtime()
    refreshed = copy.deepcopy(snapshot)
    refreshed["captured_at"] = 5.0
    refreshed["actors"][0][changed] = {
        "node_id": "other-node", "pid": 30, "create_time": 4.0,
        "endpoint": {"host": "127.0.0.1", "port": 8081},
    }[changed]
    monkeypatch.setattr(cleanup, "capture_runtime", AsyncMock(return_value=refreshed))
    monkeypatch.setattr(cleanup, "_actor_dead", AsyncMock(return_value={"dead": True}))
    monkeypatch.setattr(cleanup, "_observe_node", AsyncMock(return_value=released_node()))
    with pytest.raises(cleanup.RuntimeCleanupError, match=f"changed {changed}"):
        asyncio.run(cleanup.cleanup_runtime(replica, snapshot, timeout_s=0.1))
    replica._cleanup_runtime.assert_awaited_once()
