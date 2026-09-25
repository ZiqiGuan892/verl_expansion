"""Extend native replicas with explicit, existing-PG borrowed placement."""

import asyncio
import copy
import json
import time
from dataclasses import replace

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import get_master_addr_port
from verl.utils.device import get_device_name, get_resource_name
from verl.workers.rollout.replica import RolloutMode
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica

from multi_task_scheduler.checkpoint.checkpoint_engine_worker import MultiTaskCheckpointEngineWorker

from .http_server import MultiTaskvLLMHttpServer


class MultiTaskvLLMReplica(vLLMReplica):
    """Borrowers own new actors, never the donor PG or its existing actors."""

    def __init__(self, *args, **kwargs):
        self.allocation_kind = kwargs.pop("allocation_kind", "native")
        self.lease_id = kwargs.pop("lease_id", None)
        self.source_lease_ids = list(kwargs.pop("source_lease_ids", []))
        self.donor_task_ids = list(kwargs.pop("donor_task_ids", []))
        self.donor_replica_ranks = list(kwargs.pop("donor_replica_ranks", []))
        self.runtime_state = kwargs.pop("runtime_state", "CREATING")
        self.owns_resource_pool = bool(kwargs.pop("owns_resource_pool", self.allocation_kind == "native"))
        self.max_colocate_count = kwargs.pop("max_colocate_count", 10)
        self.claims = list(kwargs.pop("claims", []))
        self.serving_version = kwargs.pop("serving_version", None)
        self.operation_id = kwargs.pop("operation_id", None)
        if self.allocation_kind not in {"native", "borrowed"}:
            raise ValueError("allocation_kind must be 'native' or 'borrowed'")
        if self.allocation_kind == "native" and self.lease_id is not None:
            raise ValueError("native replica cannot have a lease_id")
        if self.allocation_kind == "borrowed" and self.owns_resource_pool:
            raise ValueError("borrowed replica cannot own a resource pool")
        super().__init__(*args, **kwargs)
        self.server_class = ray.remote(MultiTaskvLLMHttpServer)
        self.worker_group = None
        self.created_actor_names: list[str] = []
        self.node_layout: dict[str, dict] = {}
        self.expected_device_map: dict[int, dict] = {}
        self.actual_device_map: dict[int, dict] = {}
        self.cleanup_result: dict = {}
        self.creation_stage = "NOT_STARTED"

    def get_ray_class_with_init_args(self) -> RayClassWithInitArgs:
        return RayClassWithInitArgs(
            cls=ray.remote(MultiTaskCheckpointEngineWorker),
            rollout_config=self.config,
            model_config=self.model_config,
            replica_rank=self.replica_rank,
        )

    def _configure_borrowed_parallelism(self, spec: dict) -> None:
        """Keep borrower topology; apply only explicitly requested overrides."""
        overrides = spec.get("parallelism", {})
        if overrides:
            # RolloutConfig inherits a frozen BaseConfig. setattr on a copied
            # instance is invalid; replace constructs and validates a new one.
            self.config = replace(self.config, **overrides)
        world_size = (
            self.config.tensor_model_parallel_size
            * self.config.data_parallel_size
            * self.config.pipeline_model_parallel_size
        )
        if world_size != spec["world_size"]:
            raise ValueError("claims world_size must match borrower TP*DP*PP; provide explicit parallelism")
        self.world_size = world_size
        nodes = {claim["node_id"] for claim in self.claims}
        self.nnodes = len(nodes)
        self.gpus_per_replica_node = world_size // self.nnodes
        self.gpus_per_node = self.gpus_per_replica_node

    @staticmethod
    def _validate_claim_layout(claims: list[dict], world_size: int) -> tuple[dict, dict]:
        """Build the uniform node layout used by Worker and server startup."""
        ordered = sorted(claims, key=lambda item: item["rank"])
        if len(ordered) != world_size or [c["rank"] for c in ordered] != list(range(world_size)):
            raise ValueError("claim ranks must be contiguous and match world_size")
        keys = [(c["node_rank"], c["local_rank"]) for c in ordered]
        if keys != sorted(keys):
            raise ValueError("claims must be rank ordered by node_rank then local_rank")
        node_ranks = sorted({c["node_rank"] for c in ordered})
        if node_ranks != list(range(len(node_ranks))):
            raise ValueError("node ranks must be contiguous")
        node_ids = {c["node_id"]: c["node_rank"] for c in ordered}
        if len(node_ids) != len({c["node_rank"] for c in ordered}):
            raise ValueError("each node_id must map to exactly one node_rank")
        counts = {node_rank: sum(c["node_rank"] == node_rank for c in ordered) for node_rank in node_ids.values()}
        if len(set(counts.values())) != 1:
            raise ValueError("claims must have a uniform number of workers per node")
        node_layout = {
            node_id: {
                "node_rank": node_rank,
                "ranks": [c["rank"] for c in ordered if c["node_id"] == node_id],
                "gpu_uuids": [c["gpu_uuid"] for c in ordered if c["node_id"] == node_id],
                "local_gpu_indices": [c.get("local_gpu_index", c["local_rank"]) for c in ordered if c["node_id"] == node_id],
            }
            for node_id, node_rank in node_ids.items()
        }
        expected = {
            c["rank"]: {"node_id": c["node_id"], "gpu_uuid": c["gpu_uuid"], "local_rank": c["local_rank"]}
            for c in ordered
        }
        return node_layout, expected

    @staticmethod
    def _resolve_placement_groups(claims: list[dict]) -> dict[str, PlacementGroup]:
        """Resolve existing named PG handles without allocating a PG."""
        groups = {}
        for claim in claims:
            name = claim.get("pg_name", claim["pg_id"])
            if name not in groups:
                pg = ray.util.get_placement_group(name)
                if pg is None:
                    raise RuntimeError(
                        f"placement group {name!r} is unavailable; it must be globally named and visible"
                    )
                groups[name] = pg
            bundle_count = getattr(groups[name], "bundle_count", None)
            if bundle_count is not None and claim["bundle_index"] >= bundle_count:
                raise ValueError(f"placement group {name!r} has an invalid bundle index")
        return groups

    def validate_placement(self, spec: dict) -> dict[str, PlacementGroup]:
        """Resolve globally named PGs; Ray remains the capacity arbiter."""
        if time.time() >= spec["expires_at"]:
            raise ValueError("lease has expired")
        if self.lease_id != spec["lease_id"]:
            raise ValueError("lease_id does not own this replica")
        claims = spec["claims"]
        world_size = spec["world_size"]
        nodes = sorted({c["node_rank"] for c in claims})
        if not claims or not nodes or world_size % len(nodes):
            raise ValueError("claims must have a uniform number of workers per node")
        self._validate_claim_layout(claims, world_size)
        # Different replicas may share a card. Two TP/DP ranks in ONE engine
        # cannot count the same physical card as two accelerator devices.
        devices = {(c["node_id"], c["gpu_uuid"]) for c in claims}
        bundles = {(c["pg_id"], c["bundle_index"]) for c in claims}
        if len(devices) != world_size or len(bundles) != world_size:
            raise ValueError("one replica requires distinct devices; share bundles across replicas")

        if get_device_name() == "npu":
            # Native launch_servers concatenates Worker devices in rank order.
            # CANN requires ascending ASCEND_RT_VISIBLE_DEVICES. Reject an
            # invalid GS layout rather than changing its rank/claim identities.
            for node_rank in nodes:
                local_claims = sorted(
                    (c for c in claims if c["node_rank"] == node_rank), key=lambda c: c["local_rank"]
                )
                device_ids = [
                    str(c.get("accelerator_id", c.get("local_gpu_index")
                              if c.get("local_gpu_index") is not None else c["gpu_uuid"]))
                    for c in local_claims
                ]
                self._validate_npu_device_order(device_ids)

        return self._resolve_placement_groups(claims)

    @staticmethod
    def _validate_npu_device_order(device_ids: list[str]) -> None:
        if not device_ids or any(not str(device).isascii() or not str(device).isdecimal() for device in device_ids):
            raise ValueError(f"NPU placement requires numeric Ray accelerator IDs, got {device_ids}")
        indices = [int(device) for device in device_ids]
        if indices != sorted(set(indices)):
            raise ValueError(
                f"NPU devices must be distinct and ascending in borrower local_rank order, got {device_ids}; "
                "assign claim ranks before CE Worker creation, not just by sorting the server visibility string"
            )

    async def _create_workers_from_claims(self, groups: dict[str, PlacementGroup], spec: dict) -> None:
        """Create independent workers using native resource-option translation."""
        first = self.claims[0]
        self.creation_stage = "MASTER_ADDRESS"
        port_ref = get_master_addr_port.options(
            # This short control task binds a port, uses no accelerator, and
            # must run even when colocated actors leave less than one CPU.
            num_cpus=0,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=groups[first.get("pg_name", first["pg_id"])],
                placement_group_bundle_index=first["bundle_index"],
            ),
        ).remote()
        try:
            master_addr, master_port = await port_ref
        except asyncio.CancelledError:
            ray.cancel(port_ref, force=True)
            raise
        job_id = str(ray.get_runtime_context().get_job_id())
        prefix = f"borrowed_{job_id}_{self.replica_rank}_{self.operation_id or 'op'}_"
        self.creation_stage = "CE_WORKERS"
        for claim in self.claims:
            actor_name = f"{prefix}ce_{claim['rank']}"
            env_vars = {
                "WORLD_SIZE": str(self.world_size),
                "RANK": str(claim["rank"]),
                # A fractionally scheduled CE actor sees one device. Its CUDA
                # LOCAL_RANK defaults to 0, not the borrower's engine local rank.
                # Native Worker handles the Ascend NOSET-visible-devices case.
                "LOCAL_RANK": "0",
                "LOCAL_WORLD_SIZE": "1",
                "RAY_LOCAL_WORLD_SIZE": str(self.gpus_per_replica_node),
                "WG_PREFIX": prefix,
                "WG_BACKEND": "ray",
                "MASTER_ADDR": str(master_addr),
                "MASTER_PORT": str(master_port),
            }
            factory = self.get_ray_class_with_init_args()
            factory.update_options({
                "runtime_env": {"env_vars": env_vars},
                "name": actor_name,
                "num_cpus": claim["cpu_request"],
            })
            worker = factory(
                placement_group=groups[claim.get("pg_name", claim["pg_id"])],
                placement_group_bundle_idx=claim["bundle_index"],
                use_gpu=True,
                num_gpus=claim["gpu_fraction"],
                device_name=get_device_name(),
            )
            # Record immediately: if actor N creation fails, actors 0..N-1
            # remain available for precise cleanup.
            self.workers.append(worker)
            self.created_actor_names.append(actor_name)
        group = RayWorkerGroup.from_detached(
            name_prefix=prefix, worker_handles=self.workers,
            ray_cls_with_init=self.get_ray_class_with_init_args(),
            device_name=get_device_name(),
        )
        self.worker_group = group
        self.workers = list(group.workers)
        self.creation_stage = "CE_PLACEMENT_VALIDATION"
        await self._validate_workers()

    async def _validate_workers(self) -> None:
        def inspect_worker(worker):
            import os
            context = ray.get_runtime_context()
            return {
                "node_id": context.get_node_id(),
                "accelerator_id": str(context.get_accelerator_ids()[get_resource_name()][0]),
                "rank": int(os.environ["RANK"]),
                "world_size": int(os.environ["WORLD_SIZE"]),
                "local_world_size": int(os.environ["RAY_LOCAL_WORLD_SIZE"]),
                "actor_id": str(context.get_actor_id()),
                "pid": os.getpid(),
                "visible_devices": {
                    key: os.environ.get(key)
                    for key in ("ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
                },
            }
        infos = await asyncio.gather(*[w.__ray_call__.remote(inspect_worker) for w in self.workers])
        self.actual_device_map = dict(enumerate(infos))
        if len(infos) != self.world_size:
            raise RuntimeError("CE worker count does not match borrower world_size")
        for claim, info in zip(self.claims, infos, strict=True):
            expected_devices = {str(claim.get("accelerator_id", claim["gpu_uuid"]))}
            if claim.get("local_gpu_index") is not None:
                expected_devices.add(str(claim["local_gpu_index"]))
            if (
                info["node_id"] != claim["node_id"] or info["accelerator_id"] not in expected_devices
                or info["rank"] != claim["rank"] or info["world_size"] != self.world_size
                or info["local_world_size"] != self.gpus_per_replica_node
            ):
                raise RuntimeError(f"CE placement/rank mismatch for claim {claim['claim_id']}: {info}")
        if len({(i["node_id"], i["accelerator_id"]) for i in infos}) != self.world_size:
            raise RuntimeError("multiple engine ranks were placed on one physical accelerator")
        if get_device_name() == "npu":
            for offset in range(0, self.world_size, self.gpus_per_replica_node):
                self._validate_npu_device_order(
                    [info["accelerator_id"] for info in infos[offset : offset + self.gpus_per_replica_node]]
                )
        print(
            f"BORROWED_WORKER_PLACEMENT {json.dumps({'replica_rank': self.replica_rank, 'workers': infos})}",
            flush=True,
        )

    async def validate_runtime(self) -> dict:
        """Check server placement and the live engine's own health."""
        if len(self.servers) != self.nnodes or not self._server_address:
            raise RuntimeError("server count or serving endpoint is invalid")

        def inspect_server(server):
            import os
            context = ray.get_runtime_context()
            return {
                "node_id": context.get_node_id(), "node_rank": server.node_rank,
                "actor_id": str(context.get_actor_id()), "pid": os.getpid(),
            }
        infos = await asyncio.gather(*[s.__ray_call__.remote(inspect_server) for s in self.servers])
        for index, info in enumerate(infos):
            expected = self.claims[index * self.gpus_per_replica_node]
            if info["node_id"] != expected["node_id"] or info["node_rank"] != index:
                raise RuntimeError(f"HTTP server placement mismatch: {info}")
        # vLLMHttpServer exposes this native control-plane getter once the
        # engine has completed launch; unlike a synthetic health method it is
        # available on both the 0.9 and current native APIs.
        await self.servers[0].get_server_address.remote()
        return {
            "world_size": self.world_size,
            "server_address": self._server_address,
            "workers": copy.deepcopy(self.actual_device_map),
            "servers": infos,
        }

    async def _cleanup_runtime(self) -> dict:
        """Request termination of owned actors, never claim a lease is released."""
        # TODO(lifecycle): replace this D3/test cleanup with the production
        # destroy transaction: drain/abort requests, remove CE/LB membership,
        # release communication groups, then destroy server and worker actors.
        result = {"kill_requested": [], "errors": [], "release_confirmed": False}
        for handle in list(self.servers) + list(self.workers):
            try:
                ray.kill(handle, no_restart=True)
                result["kill_requested"].append(str(handle._actor_id))
            except Exception as exc:
                result["errors"].append(str(exc))
        # Keep handles and diagnostics even after kill requests. Actor death
        # and engine-child memory release require independent confirmation.
        self.cleanup_result = result
        return result

    async def init_from_lease(self, spec: dict) -> dict:
        """Create one bounded borrowed runtime without CE sync or LB publication."""
        if self.allocation_kind != "borrowed":
            raise ValueError("init_from_lease is valid only for borrowed replicas")
        if self.workers or self.servers:
            raise RuntimeError("replica already owns runtime actors; use manager idempotency")
        self.runtime_state = "CREATING"
        self.creation_stage = "PLACEMENT_VALIDATION"
        try:
            if not spec.get("claims") or spec.get("world_size") != len(spec["claims"]):
                raise ValueError("borrowed spec must contain one claim per world rank")
            self.claims = copy.deepcopy(spec["claims"])
            self.node_layout, self.expected_device_map = self._validate_claim_layout(
                self.claims, spec["world_size"]
            )
            self._configure_borrowed_parallelism(spec)
            groups = self.validate_placement(spec)
            timeout = min(spec.get("creation_timeout_s", 600.0), spec["expires_at"] - time.time())
            if timeout <= 0:
                raise ValueError("lease has expired")

            async def initialize():
                await self._create_workers_from_claims(groups, spec)
                self.rollout_mode = RolloutMode.STANDALONE
                self.creation_stage = "HTTP_ENGINE_START"
                await self.launch_servers()
                self.creation_stage = "RUNTIME_VALIDATION"
                return await self.validate_runtime()

            try:
                runtime = await asyncio.wait_for(initialize(), timeout=timeout)
            except asyncio.TimeoutError as exc:
                raise TimeoutError(
                    f"borrowed replica {self.replica_rank} creation timeout (limit={timeout:.1f}s) "
                    f"at stage={self.creation_stage}, lease_id={self.lease_id}: {str(exc) or repr(exc)}"
                ) from exc
            if time.time() >= spec["expires_at"]:
                raise ValueError("lease expired while starting runtime")
            self.runtime_state = "RUNTIME_READY"
            self.creation_stage = "RUNTIME_READY"
            return self.runtime_metadata(runtime)
        except (Exception, asyncio.CancelledError):
            self.runtime_state = "FAILED"
            await self._cleanup_runtime()
            raise

    def runtime_metadata(self, runtime: dict | None = None) -> dict:
        """Return diagnostics without exposing actor or PG handles."""
        result = copy.deepcopy(runtime or {})
        result.update(
            {
                "replica_rank": self.replica_rank,
                "lease_id": self.lease_id,
                "state": self.runtime_state,
                "world_size": self.world_size,
                "node_layout": copy.deepcopy(self.node_layout),
                "expected_device_map": copy.deepcopy(self.expected_device_map),
                "actual_device_map": copy.deepcopy(self.actual_device_map),
                "created_actor_names": list(self.created_actor_names),
            }
        )
        return result

    @staticmethod
    def _lifecycle_receipt(code: str, message: str, *, lease_id: str | None, state: str) -> dict:
        return {
            "operation_id": None, "lease_id": lease_id, "replica_rank": None,
            "state": state, "released": False, "error": {"code": code, "message": message},
        }

    async def destroy(self) -> dict:
        """Reserved lifecycle entry; failed-creation cleanup is not destroy."""
        # TODO(lifecycle): implement the production destroy state machine and
        # confirm that all actor, engine and communication resources are gone.
        return self._lifecycle_receipt(
            "LIFECYCLE_NOT_IMPLEMENTED", "production destroy is deferred",
            lease_id=self.lease_id, state=self.runtime_state,
        )

    async def reclaim(self, lease_id: str) -> dict:
        """Reserve reclaim and reject leases that do not own this runtime."""
        # TODO(lifecycle): implement reclaim after drain/partial-rollout,
        # CE removal/finalize, server shutdown and donor claim release.
        if self.allocation_kind != "borrowed":
            return self._lifecycle_receipt(
                "INVALID_ALLOCATION_KIND", "native replica cannot be reclaimed by borrower lease",
                lease_id=lease_id, state=self.runtime_state,
            )
        if lease_id != self.lease_id:
            return self._lifecycle_receipt(
                "LEASE_MISMATCH", "lease_id does not own this borrowed replica",
                lease_id=lease_id, state=self.runtime_state,
            )
        return self._lifecycle_receipt(
            "LIFECYCLE_NOT_IMPLEMENTED", "production reclaim is deferred",
            lease_id=lease_id, state=self.runtime_state,
        )
