"""Mocked RPC/client tests; these do not prove Ray, vLLM, or NPU generation."""

import ast
import asyncio
import copy
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/multi_task_scheduler/testing/e2e_generation.py"
spec = importlib.util.spec_from_file_location("e2e_generation_fixture_under_test", SOURCE)
generation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generation)


class Remote:
    def __init__(self, callback):
        self.callback = callback

    def remote(self, **kwargs):
        async def call():
            value = self.callback(**kwargs)
            return await value if hasattr(value, "__await__") else value

        return asyncio.create_task(call())


class MockBalancer:
    def __init__(self, routes):
        self.routes = dict(routes)
        self.inflight = {sid: 0 for sid in routes}
        self.peak = 0
        self.removed = []
        self.added = []
        self.get_all_servers = Remote(lambda: list(self.routes))
        self.get_total_inflight = Remote(lambda: sum(self.inflight.values()))
        self.remove_servers = Remote(self.remove)
        self.add_servers = Remote(self.add)
        self.clear_sticky_cache = Remote(lambda: None)

    def remove(self, server_ids):
        self.removed.extend(server_ids)
        for sid in server_ids:
            del self.routes[sid]
            del self.inflight[sid]

    def add(self, servers):
        self.added.extend(servers)
        for sid, handle in servers.items():
            self.routes[sid] = handle
            self.inflight[sid] = 0


class MockClient:
    """Explicit native-client substitute that performs mocked acquire/release."""

    def __init__(self, lb):
        self.lb = lb

    async def _acquire_server(self, request_id, **extra):
        assert len(self.lb.routes) == 1
        sid = next(iter(self.lb.routes))
        self.lb.inflight[sid] += 1
        self.lb.peak = max(self.lb.peak, sum(self.lb.inflight.values()))
        return sid, self.lb.routes[sid]

    async def generate(self, request_id, **kwargs):
        sid, server = await self._acquire_server(request_id, **kwargs)
        try:
            return await server.generate(request_id=request_id, **kwargs)
        finally:
            # Like the native client, do not await the release operation.
            async def release():
                self.lb.inflight[sid] -= 1

            asyncio.create_task(release())


@pytest.fixture
def runtime(monkeypatch):
    module = ModuleType("verl.workers.rollout.llm_server")
    module.FullyAsyncLLMServerClient = MockClient
    monkeypatch.setitem(sys.modules, module.__name__, module)

    async def generate(**kwargs):
        await asyncio.sleep(0)
        return SimpleNamespace(
            token_ids=[31, 32],
            stop_reason="length",
            extra_fields={"global_steps": 3, "min_global_steps": 3, "max_global_steps": 3},
        )

    handles = {"native": SimpleNamespace(generate=generate), "borrowed": SimpleNamespace(generate=generate)}
    lb = MockBalancer(handles)
    manager = SimpleNamespace(
        global_load_balancer=lb,
        server_addresses=list(handles),
        server_handles=list(handles.values()),
        rollout_replicas=[SimpleNamespace(replica_rank=rank, _server_address=sid) for rank, sid in enumerate(handles)],
        get_client=lambda client_cls: client_cls(lb),
    )
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: [11, 12])
    return manager, tokenizer, lb


def test_probe_observes_each_real_route_and_restores_pool_with_mocked_client(runtime):
    manager, tokenizer, lb = runtime
    original = dict(lb.routes)
    receipt = asyncio.run(generation.probe_replicas(manager, tokenizer, [0, 1], 3, concurrency=3))
    assert receipt["state"] == "GENERATED"
    assert receipt["concurrency"] == 3
    assert receipt["inflight_before"] == receipt["inflight_after"] == 0
    assert receipt["routes_restored"] is True
    assert lb.routes == original
    assert lb.peak == 3
    assert [item["server_id"] for item in receipt["replicas"]] == ["native", "borrowed"]
    for replica in receipt["replicas"]:
        assert len(replica["requests"]) == 3
        for request in replica["requests"]:
            assert request["acquired_server_ids"] == [replica["server_id"]]
            assert request["token_count"] == 2
            assert request["min_version"] == request["max_version"] == request["global_steps"] == 3


@pytest.mark.parametrize("failure", ["empty", "version", "abort", "exception", "timeout"])
def test_probe_failure_restores_routes_and_never_reports_generated(runtime, failure):
    manager, tokenizer, lb = runtime
    original = dict(lb.routes)

    async def invalid(**kwargs):
        if failure == "exception":
            raise RuntimeError("engine failed")
        if failure == "timeout":
            await asyncio.Event().wait()
        return SimpleNamespace(
            token_ids=[] if failure == "empty" else [31],
            stop_reason="aborted" if failure == "abort" else "length",
            extra_fields={
                "global_steps": 2 if failure == "version" else 3,
                "min_global_steps": 3,
                "max_global_steps": 3,
            },
        )

    lb.routes["borrowed"].generate = invalid
    error = asyncio.TimeoutError if failure == "timeout" else RuntimeError
    with pytest.raises(error):
        asyncio.run(generation.probe_replicas(manager, tokenizer, [1], 3, concurrency=2, timeout_seconds=0.05))
    assert lb.routes == original
    assert sum(lb.inflight.values()) == 0


def test_probe_rejects_busy_lb_without_mutation(runtime):
    manager, tokenizer, lb = runtime
    lb.inflight["native"] = 1
    with pytest.raises(RuntimeError, match="requires an idle LB"):
        asyncio.run(generation.probe_replicas(manager, tokenizer, [1], 3))
    assert not lb.removed and not lb.added


def test_probe_refuses_unknown_routes_that_cannot_be_restored(runtime):
    manager, tokenizer, lb = runtime
    lb.routes["unknown"] = object()
    lb.inflight["unknown"] = 0
    with pytest.raises(RuntimeError, match="unknown manager handles"):
        asyncio.run(generation.probe_replicas(manager, tokenizer, [1], 3))
    assert not lb.removed and not lb.added


def test_probe_accepts_manager_owned_borrowed_record_before_projection(runtime):
    manager, tokenizer, lb = runtime
    borrowed = manager.rollout_replicas.pop()
    manager._borrowed_record_by_rank = lambda rank: {"replica": borrowed}
    receipt = asyncio.run(generation.probe_replicas(manager, tokenizer, [1], 3))
    assert receipt["replicas"][0]["server_id"] == "borrowed"
    assert set(lb.routes) == {"native", "borrowed"}


def _server_class(parent):
    path = ROOT / "src/multi_task_scheduler/rollout/http_server.py"
    parsed = ast.parse(path.read_text())
    node = next(item for item in parsed.body if isinstance(item, ast.ClassDef))
    scope = {"vLLMHttpServer": parent, "copy": copy}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope[node.name]


def test_audit_is_disabled_by_default_and_preserves_native_output_object():
    output = SimpleNamespace(token_ids=[1], stop_reason="length", extra_fields={"global_steps": 3})

    class NativeSubstitute:
        async def generate(self, *args, **kwargs):
            assert args == ("prompt",) and kwargs == {"sampling": "value"}
            return output

    server = _server_class(NativeSubstitute)()
    assert asyncio.run(server.generate("prompt", sampling="value")) is output
    assert server.get_test_generation_audit() == {"enabled": False}
    asyncio.run(server.set_test_generation_audit())
    assert asyncio.run(server.generate("prompt", sampling="value")) is output
    audit = server.get_test_generation_audit()
    assert audit["started_calls"] == audit["successful_calls"] == audit["nonempty_calls"] == 1
    assert audit["failed_calls"] == audit["inflight"] == 0
    assert audit["token_count"] == audit["peak_inflight"] == 1
    assert audit["calls_by_version"] == audit["nonempty_by_version"] == {"3": 1}
    audit["calls_by_version"]["3"] = 100
    assert server.get_test_generation_audit()["calls_by_version"] == {"3": 1}
    asyncio.run(server.set_test_generation_audit(enabled=False, reset=False))
    asyncio.run(server.generate("prompt", sampling="value"))
    assert server.get_test_generation_audit()["started_calls"] == 1


def test_audit_counts_native_failures_and_aborts_without_hiding_them():
    class NativeSubstitute:
        async def generate(self, fail=False):
            if fail:
                raise RuntimeError("native error")
            return SimpleNamespace(token_ids=[1], stop_reason="aborted", extra_fields={"global_steps": 4})

    server = _server_class(NativeSubstitute)()
    asyncio.run(server.set_test_generation_audit())
    asyncio.run(server.generate())
    with pytest.raises(RuntimeError, match="native error"):
        asyncio.run(server.generate(fail=True))
    audit = server.get_test_generation_audit()
    assert audit["started_calls"] == 2
    assert audit["successful_calls"] == audit["failed_calls"] == 1
    assert audit["nonempty_calls"] == audit["inflight"] == 0
    assert audit["calls_by_version"] == {"4": 1}
    assert audit["nonempty_by_version"] == {}


def test_audit_tracks_concurrent_calls_and_cancellation_and_rejects_live_reset():
    async def run():
        gate = asyncio.Event()

        class NativeSubstitute:
            async def generate(self):
                await gate.wait()
                return SimpleNamespace(token_ids=[1, 2], stop_reason="stop", extra_fields={"global_steps": 4})

        server = _server_class(NativeSubstitute)()
        await server.set_test_generation_audit()
        first = asyncio.create_task(server.generate())
        second = asyncio.create_task(server.generate())
        await asyncio.sleep(0)
        with pytest.raises(RuntimeError, match="only be changed while idle"):
            await server.set_test_generation_audit()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        gate.set()
        await second
        audit = server.get_test_generation_audit()
        assert audit["peak_inflight"] == audit["started_calls"] == 2
        assert audit["successful_calls"] == audit["failed_calls"] == audit["nonempty_calls"] == 1
        assert audit["token_count"] == 2
        assert audit["inflight"] == 0

    asyncio.run(run())
