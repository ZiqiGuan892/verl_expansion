"""D2 runtime-planning tests that do not start Ray or vLLM.

The tests exercise the claim layout and borrowed parallelism decisions in the
real extension class body.  Actor placement, engine startup and device
identity remain GPU acceptance tests because they require a live Ray cluster.
"""

import ast
import asyncio
import copy
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest


SOURCE = Path(__file__).resolve().parents[2] / "src/multi_task_scheduler/rollout/replica.py"


@dataclass(frozen=True)
class _Config:
    tensor_model_parallel_size: int = 4
    data_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    expert_parallel_size: int = 1
    moe_tensor_parallel_size: int = 1


class _Parent:
    def __init__(self, *args, **kwargs):
        self.replica_rank = kwargs.get("replica_rank", 0)
        self.world_size = 4
        self.config = _Config()
        self.model_config = object()
        self.workers = []
        self.servers = []


def _replica_class():
    parsed = ast.parse(SOURCE.read_text())
    node = next(item for item in parsed.body if isinstance(item, ast.ClassDef) and item.name == "MultiTaskvLLMReplica")
    node.bases = [ast.Name(id="TestParent", ctx=ast.Load())]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )
    ray = SimpleNamespace(remote=Mock(return_value=object()))
    scope = {
        "TestParent": _Parent,
        "asyncio": asyncio,
        "copy": copy,
        "json": json,
        "time": time,
        "replace": replace,
        "ray": ray,
        "RayClassWithInitArgs": object,
        "RayWorkerGroup": object,
        "PlacementGroupSchedulingStrategy": object,
        "get_platform": lambda: object(),
        "get_master_addr_port": object(),
        "get_device_name": lambda: "cpu",
        "get_resource_name": lambda: "GPU",
        "RolloutMode": SimpleNamespace(STANDALONE="standalone"),
        "MultiTaskCheckpointEngineWorker": object,
        "MultiTaskvLLMHttpServer": object,
    }
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), scope)
    return scope["MultiTaskvLLMReplica"]


def _claim(rank, node_rank, local_rank, node_id=None):
    return {
        "rank": rank,
        "node_rank": node_rank,
        "local_rank": local_rank,
        "node_id": node_id or f"node-{node_rank}",
        "gpu_uuid": f"gpu-{rank}",
    }


def test_claim_layout_is_rank_ordered_and_serializable():
    replica = _replica_class()(replica_rank=0, allocation_kind="borrowed", owns_resource_pool=False)
    layout, expected = replica._validate_claim_layout(
        [_claim(0, 0, 0), _claim(1, 0, 1), _claim(2, 1, 0), _claim(3, 1, 1)], 4
    )
    assert layout["node-0"]["ranks"] == [0, 1]
    assert layout["node-1"]["ranks"] == [2, 3]
    assert expected[3]["gpu_uuid"] == "gpu-3"


def test_claim_layout_rejects_interleaved_node_groups():
    replica = _replica_class()(replica_rank=0, allocation_kind="borrowed", owns_resource_pool=False)
    with pytest.raises(ValueError, match="rank ordered"):
        replica._validate_claim_layout(
            [_claim(0, 0, 0), _claim(1, 1, 0), _claim(2, 0, 1), _claim(3, 1, 1)], 4
        )


def test_borrowed_parallelism_copies_config_for_heterogeneous_world_size():
    replica = _replica_class()(replica_rank=0, allocation_kind="borrowed", owns_resource_pool=False)
    original = replica.config
    replica.claims = [_claim(0, 0, 0), _claim(1, 0, 1)]
    replica._configure_borrowed_parallelism(
        {"world_size": 2, "parallelism": {"tensor_model_parallel_size": 2, "data_parallel_size": 1,
                                             "pipeline_model_parallel_size": 1}}
    )
    assert replica.world_size == 2
    assert replica.nnodes == 1
    assert replica.gpus_per_replica_node == 2
    assert replica.config is not original
    assert replica.config.tensor_model_parallel_size == 2
    assert replica.config.data_parallel_size == 1
    assert original.tensor_model_parallel_size == 4


def test_placement_group_resolution_requires_named_visible_groups():
    class _Util:
        @staticmethod
        def get_placement_group(name):
            return {"pg-a": "handle-a"}.get(name)

    replica_class = _replica_class()
    replica_class.__init__.__globals__["ray"].util = _Util
    with pytest.raises(RuntimeError, match="globally named"):
        replica_class._resolve_placement_groups([{"pg_id": "missing", "bundle_index": 0}])
    assert replica_class._resolve_placement_groups([{"pg_id": "pg-a", "bundle_index": 0}]) == {"pg-a": "handle-a"}


def test_second_shared_bundle_worker_can_start_with_half_cpu_remaining():
    """Donor(1 CPU) + borrower A(0.5 CPU) leave only 0.5 of a 2 CPU bundle."""
    replica_class = _replica_class()
    scope = replica_class.__init__.__globals__
    port_options = {}

    class PortTask:
        def options(self, **kwargs):
            port_options.update(kwargs)
            return self

        async def remote(self):
            # A native Ray task defaults to 1 CPU and cannot run here.
            assert port_options.get("num_cpus", 1) <= 0.5
            return "127.0.0.1", 12345

    scope["get_master_addr_port"] = PortTask()
    scope["PlacementGroupSchedulingStrategy"] = lambda **kwargs: SimpleNamespace(**kwargs)
    scope["ray"].get_runtime_context = lambda: SimpleNamespace(get_job_id=lambda: "job")
    scope["RayWorkerGroup"] = SimpleNamespace(
        from_detached=lambda **kwargs: SimpleNamespace(workers=kwargs["worker_handles"])
    )
    worker = object()
    factory = Mock(return_value=worker)
    replica = replica_class(replica_rank=2, allocation_kind="borrowed", lease_id="lease-b")
    replica.world_size = replica.gpus_per_replica_node = 1
    replica.claims = [{**_claim(0, 0, 0), "pg_id": "pg-a", "bundle_index": 3,
                       "cpu_request": 0.5, "gpu_fraction": 0.25}]
    replica.get_ray_class_with_init_args = Mock(return_value=factory)
    replica._validate_workers = AsyncMock()
    pg = object()

    asyncio.run(replica._create_workers_from_claims({"pg-a": pg}, {}))

    assert port_options["num_cpus"] == 0
    assert port_options["scheduling_strategy"].placement_group is pg
    assert port_options["scheduling_strategy"].placement_group_bundle_index == 3
    assert factory.update_options.call_args.args[0]["num_cpus"] == 0.5
    assert factory.call_args.kwargs["num_gpus"] == 0.25
    assert factory.call_args.kwargs["placement_group"] is pg
    assert factory.call_args.kwargs["placement_group_bundle_idx"] == 3
    assert replica.workers == [worker]
    replica._validate_workers.assert_awaited_once()


@pytest.mark.parametrize("devices", [["4", "6", "5", "7"], ["0", "0"], ["npu-4"], []])
def test_npu_device_order_rejects_invalid_visibility_before_server_launch(devices):
    with pytest.raises(ValueError, match="NPU"):
        _replica_class()._validate_npu_device_order(devices)


def test_npu_device_order_uses_numeric_order_and_allows_noncontiguous_devices():
    _replica_class()._validate_npu_device_order(["2", "4", "10"])


@pytest.mark.parametrize("devices,valid", [([4, 5, 6, 7], True), ([4, 6, 5, 7], False)])
def test_npu_placement_checks_rank_order_with_uuid_and_numeric_index(devices, valid):
    replica_class = _replica_class()
    replica_class.__init__.__globals__["get_device_name"] = lambda: "npu"
    replica = replica_class(replica_rank=2, allocation_kind="borrowed", lease_id="lease")
    replica._resolve_placement_groups = Mock(return_value={"pg": object()})
    spec = {
        "lease_id": "lease", "world_size": 4, "expires_at": time.time() + 60,
        "claims": [{**_claim(rank, 0, rank), "pg_id": "pg", "bundle_index": rank,
                    "local_gpu_index": device} for rank, device in enumerate(devices)],
    }
    if valid:
        assert replica.validate_placement(spec) is replica._resolve_placement_groups.return_value
    else:
        with pytest.raises(ValueError, match="ascending"):
            replica.validate_placement(spec)
        replica._resolve_placement_groups.assert_not_called()


def test_creation_timeout_reports_stage_and_requests_cleanup():
    replica = _replica_class()(replica_rank=2, allocation_kind="borrowed", lease_id="lease-b")
    replica.validate_placement = Mock(return_value={"pg-a": object()})
    replica._cleanup_runtime = AsyncMock(return_value={"kill_requested": [], "errors": []})

    async def pending_port(*args):
        replica.creation_stage = "MASTER_ADDRESS"
        await asyncio.Future()

    replica._create_workers_from_claims = pending_port
    spec = {
        "lease_id": "lease-b", "world_size": 1, "claims": [_claim(0, 0, 0)],
        "parallelism": {"tensor_model_parallel_size": 1},
        "expires_at": time.time() + 60, "creation_timeout_s": 0.01,
    }
    with pytest.raises(TimeoutError, match="MASTER_ADDRESS"):
        asyncio.run(replica.init_from_lease(spec))
    assert replica.runtime_state == "FAILED"
    replica._cleanup_runtime.assert_awaited_once()
