"""D2 runtime-planning tests that do not start Ray or vLLM.

The tests exercise the claim layout and borrowed parallelism decisions in the
real extension class body.  Actor placement, engine startup and device
identity remain GPU acceptance tests because they require a live Ray cluster.
"""

import ast
import copy
import inspect
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

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
        "asyncio": __import__("asyncio"),
        "copy": copy,
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


def test_http_server_shutdown_prefers_async_shutdown_and_clears_engine():
    source = Path(__file__).resolve().parents[2] / "src/multi_task_scheduler/rollout/http_server.py"
    parsed = ast.parse(source.read_text())
    node = next(item for item in parsed.body if isinstance(item, ast.ClassDef) and item.name == "MultiTaskvLLMHttpServer")
    node.bases = [ast.Name(id="TestParent", ctx=ast.Load())]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )

    class _Parent:
        pass

    class _Engine:
        def __init__(self):
            self.calls = 0

        async def shutdown(self):
            self.calls += 1

    scope = {"TestParent": _Parent, "inspect": inspect, "os": __import__("os")}
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), scope)
    server = scope["MultiTaskvLLMHttpServer"]()
    server.engine = _Engine()

    result = __import__("asyncio").run(server.shutdown_engine())
    assert result == {"shutdown": True, "method": "shutdown"}
    assert server.engine is None
