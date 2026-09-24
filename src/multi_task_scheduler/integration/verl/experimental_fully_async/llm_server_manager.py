"""Select rollout subclasses and create borrowed runtimes from GS claims."""

import asyncio
import copy
import json
import math
import time

import ray

from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncLLMServerManager
from verl.workers.rollout.router import DEFAULT_ROUTING_CACHE_SIZE
from verl.utils.device import get_device_name

from multi_task_scheduler.rollout.load_balancer import MultiTaskGlobalRequestLoadBalancer
from multi_task_scheduler.rollout.replica import MultiTaskvLLMReplica


class MultiTaskLLMServerManager(FullyAsyncLLMServerManager):
    """Ordinary object owned by Rollouter; native replica lists remain authoritative."""

    def __init__(
        self,
        config,
        worker_group=None,
        rollout_resource_pool=None,
        start_rank=0,
        load_balancer_cls=None,
        *,
        group_scheduler=None,
    ):
        self.group_scheduler = group_scheduler
        # LLMServerManager explicitly preserves a preselected replica class.
        self.rollout_replica_class = MultiTaskvLLMReplica
        # FullyAsyncLLMServerManager in the paired native commit exposes only
        # the three historical constructor arguments.  Set the two native
        # manager knobs after its constructor rather than assuming the newer
        # LLMServerManager signature.
        super().__init__(config, worker_group, rollout_resource_pool)
        self.start_rank = start_rank
        self._load_balancer_cls = load_balancer_cls or MultiTaskGlobalRequestLoadBalancer

        # D1 identity and operation state.  These are local manager tables,
        # not a second global lease registry; GS remains the owner of claims.
        self.max_colocate_count = self._read_max_colocate_count(config)
        self.next_replica_rank = int(start_rank)
        self.retired_replica_ranks: set[int] = set()
        self.borrowed_operations: dict[str, dict] = {}
        self.replica_operation_lock = asyncio.Lock()

    async def _init_global_load_balancer(self) -> None:
        # Native code forwards full_determinism only to its exact default class.
        # Our subclass keeps native routing, so it must receive the same flag.
        self.global_load_balancer = ray.remote(self._load_balancer_cls).remote(
            servers=dict(zip(self.server_addresses, self.server_handles, strict=True)),
            max_cache_size=DEFAULT_ROUTING_CACHE_SIZE,
            full_determinism=getattr(self.rollout_config, "full_determinism", False),
            group_scheduler=self.group_scheduler,
        )

    def _borrowed_record_by_rank(self, replica_rank: int) -> dict:
        for record in self.borrowed_operations.values():
            if record.get("replica_rank") == replica_rank:
                return record
        raise KeyError(f"unknown borrowed replica_rank: {replica_rank}")

    def _local_native_donors(self, spec: dict) -> list:
        """Return native replicas in this task that own the requested claims.

        GS normally coordinates a donor task separately.  The D2/D3 runtime
        smoke path uses native replicas created by this same manager, so it can
        safely put those donors to sleep before launching a second engine on
        their physical devices.  A borrower task must never guess or mutate a
        foreign task's replicas.
        """
        requested = {int(claim["donor_replica_rank"]) for claim in spec["claims"]}
        donors = [
            replica
            for replica in self.rollout_replicas
            if getattr(replica, "allocation_kind", "native") == "native"
            and int(getattr(replica, "replica_rank", -1)) in requested
        ]
        by_rank = {int(replica.replica_rank): replica for replica in donors}
        missing = sorted(requested - set(by_rank))
        if missing:
            raise RuntimeError(
                "D2/D3 local smoke requires native donor replicas for ranks "
                f"{missing}; cross-task donors must be slept by their owner TaskRunner"
            )
        return [by_rank[rank] for rank in sorted(requested)]

    @staticmethod
    async def _test_memory_call(replica, method: str, *args) -> None:
        """Call an opt-in runtime fixture on every node of one test replica."""
        await asyncio.gather(*(getattr(server, method).remote(*args) for server in replica.servers))

    async def get_replica_for_ce(self, replica_rank: int):
        """Return the local borrowed replica projection for Trainer CE wiring."""
        if isinstance(replica_rank, bool) or not isinstance(replica_rank, int) or replica_rank < 0:
            raise ValueError("replica_rank must be a non-negative integer")
        record = self._borrowed_record_by_rank(replica_rank)
        replica = record.get("replica")
        if replica is None or record.get("state") != "RUNTIME_READY":
            raise RuntimeError(f"borrowed replica {replica_rank} is not RUNTIME_READY")
        return replica

    async def register_borrowed_replica_for_ce(self, replica_rank: int) -> dict:
        """Expose a borrowed runtime to local replica projections without LB publication."""
        replica = await self.get_replica_for_ce(replica_rank)
        if replica not in self.rollout_replicas:
            self.rollout_replicas.append(replica)
        return {"replica_rank": replica_rank, "state": "RUNTIME_REGISTERED"}

    async def mark_replica_serving_version(self, replica_rank: int, version: int) -> dict:
        """Persist CE's confirmed version on the manager-owned replica object."""
        if isinstance(version, bool) or not isinstance(version, int) or version < 0:
            raise ValueError("version must be a non-negative integer")
        replica = await self.get_replica_for_ce(replica_rank)
        replica.serving_version = version
        return {"replica_rank": replica_rank, "serving_version": version}

    async def cleanup_d3_runtime(self, replica_rank: int, global_steps: int | None = None) -> dict:
        """Clean only the temporary D3 borrowed actors after CE checks."""
        record = self._borrowed_record_by_rank(replica_rank)
        replica = record.get("replica")
        if replica is None:
            return {"replica_rank": replica_rank, "state": "NOT_FOUND"}
        # Wait for device allocations to be released before restoring donor
        # memory. Killing Ray actors alone does not acknowledge that release.
        await self._test_memory_call(replica, "sleep_for_runtime_test")
        cleanup = await replica._cleanup_runtime()
        for donor in record.get("sleeping_donors", []):
            await self._test_memory_call(donor, "wake_for_runtime_test")
            if global_steps is not None:
                # Donors were excluded from the CE topology while the
                # borrowed worker occupied their physical slots.  Restore the
                # serving tag explicitly before generation resumes.
                await self._test_memory_call(donor, "set_global_steps", int(global_steps))
        record["sleeping_donors"] = []
        self.rollout_replicas = [item for item in self.rollout_replicas if item is not replica]
        record["state"] = "DESTROYED"
        record["cleanup"] = copy.deepcopy(cleanup)
        return {"replica_rank": replica_rank, "state": "DESTROYED", "cleanup": cleanup}

    @staticmethod
    def _read_max_colocate_count(config) -> int:
        """Read M without requiring a particular OmegaConf implementation."""
        rollout = getattr(getattr(config, "actor_rollout_ref", None), "rollout", None)
        value = None
        if rollout is not None:
            try:
                value = rollout.get("max_colocate_count", None)
            except AttributeError:
                value = getattr(rollout, "max_colocate_count", None)
        if value is None:
            value = 10
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("max_colocate_count must be a positive integer")
        return value

    @staticmethod
    def _copy_mapping(value, name: str) -> dict:
        if not isinstance(value, dict):
            raise ValueError(f"{name} must be a mapping")
        return copy.deepcopy(value)

    @staticmethod
    def _required_string(mapping: dict, key: str, *, name: str | None = None) -> str:
        value = mapping.get(key)
        label = name or key
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must be a non-empty string")
        return value

    @staticmethod
    def _required_non_negative_int(mapping: dict, key: str, *, name: str | None = None) -> int:
        value = mapping.get(key)
        label = name or key
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{label} must be a non-negative integer")
        return value

    @staticmethod
    def _required_positive_number(mapping: dict, key: str, *, upper: float | None = None) -> float:
        value = mapping.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be a finite positive number")
        value = float(value)
        if upper is not None and value > upper:
            raise ValueError(f"{key} must be <= {upper}")
        return value

    def _validate_create_spec(self, spec: dict) -> dict:
        """Validate and freeze a GS placement request before taking the lock.

        This method performs only deterministic contract checks.  It does not
        resolve a PlacementGroup, query Ray, reserve capacity, or create an
        actor; those checks belong to D2.
        """
        source = self._copy_mapping(spec, "spec")
        operation_id = self._required_string(source, "operation_id")
        lease_id = self._required_string(source, "lease_id")
        borrower_task_id = self._required_string(source, "borrower_task_id")
        borrower_replica_id = self._required_string(source, "borrower_replica_id")

        world_size = self._required_non_negative_int(source, "world_size")
        if world_size <= 0:
            raise ValueError("world_size must be greater than zero")
        placement_epoch = self._required_non_negative_int(source, "placement_epoch")
        expires_at = source.get("expires_at")
        if isinstance(expires_at, bool) or not isinstance(expires_at, (int, float)):
            raise ValueError("expires_at must be a Unix timestamp")
        if not math.isfinite(float(expires_at)) or float(expires_at) <= time.time():
            raise ValueError("lease has expired")

        max_colocate_count = source.get("max_colocate_count", self.max_colocate_count)
        if isinstance(max_colocate_count, bool) or not isinstance(max_colocate_count, int) or max_colocate_count <= 0:
            raise ValueError("max_colocate_count must be a positive integer")

        raw_claims = source.get("claims", source.get("selected_slots"))
        if not isinstance(raw_claims, list) or not raw_claims:
            raise ValueError("claims or selected_slots must be a non-empty list")
        if len(raw_claims) != world_size:
            raise ValueError("world_size must equal the number of claims")

        requested_rank = source.get("replica_rank")
        if requested_rank is not None:
            if isinstance(requested_rank, bool) or not isinstance(requested_rank, int) or requested_rank < 0:
                raise ValueError("replica_rank must be a non-negative integer or null")

        claims: list[dict] = []
        claim_ids: set[str] = set()
        source_lease_ids: list[str] = []
        node_claims: dict[str, list[dict]] = {}
        bundle_usage: dict[tuple[str, int], dict[str, float]] = {}
        for index, raw_claim in enumerate(raw_claims):
            claim = self._copy_mapping(raw_claim, f"claims[{index}]")
            claim_id = self._required_string(claim, "claim_id")
            if claim_id in claim_ids:
                raise ValueError(f"duplicate claim_id: {claim_id}")
            claim_ids.add(claim_id)

            # A selected_slots list is already rank ordered.  Explicit ranks
            # are still accepted, but must form the same contiguous sequence.
            rank = claim.get("rank", index)
            if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
                raise ValueError(f"claims[{index}].rank must be a non-negative integer")
            claim["rank"] = rank

            for key in ("donor_task_id", "pg_id", "node_id", "gpu_uuid"):
                self._required_string(claim, key, name=f"claims[{index}].{key}")
            claim_lease_id = self._required_string(claim, "lease_id", name=f"claims[{index}].lease_id")
            donor_rank = self._required_non_negative_int(claim, "donor_replica_rank")
            bundle_index = self._required_non_negative_int(claim, "bundle_index")
            node_rank = self._required_non_negative_int(claim, "node_rank")
            local_rank = self._required_non_negative_int(claim, "local_rank")
            gpu_fraction = self._required_positive_number(claim, "gpu_fraction", upper=1.0)
            cpu_request = self._required_positive_number(claim, "cpu_request", upper=float(max_colocate_count))

            claim["donor_replica_rank"] = donor_rank
            claim["bundle_index"] = bundle_index
            claim["node_rank"] = node_rank
            claim["local_rank"] = local_rank
            claim["gpu_fraction"] = gpu_fraction
            claim["cpu_request"] = cpu_request
            claims.append(claim)
            node_claims.setdefault(claim["node_id"], []).append(claim)
            source_lease_ids.append(claim_lease_id)
            usage = bundle_usage.setdefault((claim["pg_id"], bundle_index), {"gpu": 0.0, "cpu": 0.0})
            usage["gpu"] += gpu_fraction
            usage["cpu"] += cpu_request

        ranks = sorted(claim["rank"] for claim in claims)
        if ranks != list(range(world_size)):
            raise ValueError("claim ranks must be a contiguous range starting at zero")
        for (pg_id, bundle_index), usage in bundle_usage.items():
            if usage["gpu"] > 1.0 + 1e-9 or usage["cpu"] > float(max_colocate_count) + 1e-9:
                raise ValueError(f"bundle capacity exceeded for {pg_id}[{bundle_index}]")

        node_ranks = sorted({claim["node_rank"] for claim in claims})
        if node_ranks != list(range(len(node_ranks))):
            raise ValueError("node_rank must be contiguous and unique per node")
        node_rank_by_id: dict[str, int] = {}
        for claim in claims:
            node_id = claim["node_id"]
            previous_rank = node_rank_by_id.setdefault(node_id, claim["node_rank"])
            if previous_rank != claim["node_rank"]:
                raise ValueError(f"node_id {node_id} maps to multiple node_rank values")
        if len(set(node_rank_by_id.values())) != len(node_rank_by_id):
            raise ValueError("each node_id must have a unique node_rank")
        counts = {node_id: len(items) for node_id, items in node_claims.items()}
        if len(set(counts.values())) != 1:
            raise ValueError("claims must have a uniform number of workers per node")
        for node_id, items in node_claims.items():
            local_ranks = sorted(claim["local_rank"] for claim in items)
            if local_ranks != list(range(len(items))):
                raise ValueError(f"local_rank must be contiguous for node {node_id}")

        explicit_lease_ids = source.get("lease_ids", source.get("source_lease_ids", source_lease_ids))
        if not isinstance(explicit_lease_ids, list) or not all(
            isinstance(item, str) and item for item in explicit_lease_ids
        ):
            raise ValueError("lease_ids must be a non-empty list of strings")
        lease_ids = list(dict.fromkeys(explicit_lease_ids))
        missing = set(source_lease_ids) - set(lease_ids)
        if missing:
            raise ValueError(f"lease_ids do not cover claim leases: {sorted(missing)}")

        raw_parallelism = source.get("parallelism")
        if raw_parallelism is not None and not isinstance(raw_parallelism, dict):
            raise ValueError("parallelism must be a mapping when provided")
        parallelism = copy.deepcopy(raw_parallelism or {})
        for key in ("tensor_model_parallel_size", "data_parallel_size", "pipeline_model_parallel_size"):
            if key in parallelism:
                value = parallelism[key]
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValueError(f"parallelism.{key} must be a positive integer")
        parallelism.setdefault("tensor_model_parallel_size", world_size)
        parallelism.setdefault("data_parallel_size", 1)
        parallelism.setdefault("pipeline_model_parallel_size", 1)
        if (
            parallelism["tensor_model_parallel_size"]
            * parallelism["data_parallel_size"]
            * parallelism["pipeline_model_parallel_size"]
            != world_size
        ):
            raise ValueError("parallelism TP*DP*PP must equal world_size")
        creation_timeout_s = source.get("creation_timeout_s", 600.0)
        if (
            isinstance(creation_timeout_s, bool)
            or not isinstance(creation_timeout_s, (int, float))
            or not math.isfinite(float(creation_timeout_s))
            or float(creation_timeout_s) <= 0
        ):
            raise ValueError("creation_timeout_s must be a finite positive number")

        return {
            "operation_id": operation_id,
            "lease_id": lease_id,
            "lease_ids": lease_ids,
            "source_lease_ids": lease_ids.copy(),
            "borrower_task_id": borrower_task_id,
            "borrower_replica_id": borrower_replica_id,
            "replica_rank": requested_rank,
            "claims": sorted(claims, key=lambda item: item["rank"]),
            "world_size": world_size,
            "max_colocate_count": max_colocate_count,
            "expires_at": float(expires_at),
            "placement_epoch": placement_epoch,
            "parallelism": parallelism,
            "creation_timeout_s": float(creation_timeout_s),
        }

    def _used_replica_ranks(self) -> set[int]:
        """Return every rank that must never be allocated again."""
        used = set(self.retired_replica_ranks)
        for replica in getattr(self, "rollout_replicas", []):
            rank = getattr(replica, "replica_rank", None)
            if isinstance(rank, int):
                used.add(rank)
        for mapping_name in ("hybrid_replicas", "alive_replicas"):
            for replica in getattr(self, mapping_name, {}).values():
                rank = getattr(replica, "replica_rank", None)
                if isinstance(rank, int):
                    used.add(rank)
        for record in self.borrowed_operations.values():
            rank = record.get("replica_rank")
            if isinstance(rank, int):
                used.add(rank)
        return used

    def _allocate_replica_rank_locked(self, requested_rank: int | None = None) -> int:
        """Allocate a monotonically increasing task-local rank; caller holds lock."""
        used = self._used_replica_ranks()
        if requested_rank is not None:
            if requested_rank in used:
                raise ValueError(f"replica_rank {requested_rank} is already in use")
            self.next_replica_rank = max(self.next_replica_rank, requested_rank + 1)
            return requested_rank
        candidate = max(0, self.next_replica_rank)
        while candidate in used:
            candidate += 1
        self.next_replica_rank = candidate + 1
        return candidate

    def _receipt(self, record: dict) -> dict:
        """Return only serializable operation metadata; never leak handles."""
        if record.get("result") is not None:
            return copy.deepcopy(record["result"])
        return {
            "operation_id": record["operation_id"],
            "lease_id": record["lease_id"],
            "lease_ids": list(record["source_lease_ids"]),
            "replica_rank": record["replica_rank"],
            "state": record["state"],
            "released": False,
            "error": {"code": "OPERATION_IN_PROGRESS", "message": "borrowed runtime creation is still running"},
        }

    @staticmethod
    def _same_request(record: dict, spec: dict) -> bool:
        previous = copy.deepcopy(record.get("request_spec"))
        current = copy.deepcopy(spec)
        # operation_id identifies a caller attempt; lease_id plus the frozen
        # placement contract identifies the idempotent create request.
        previous_rank = previous.get("replica_rank") if isinstance(previous, dict) else None
        current_rank = current.get("replica_rank") if isinstance(current, dict) else None
        if previous_rank is None:
            assigned_rank = record.get("replica_rank")
            if current_rank is not None and current_rank != assigned_rank:
                return False
        elif current_rank is not None and previous_rank != current_rank:
            return False
        if isinstance(previous, dict):
            previous.pop("operation_id", None)
            previous.pop("replica_rank", None)
        if isinstance(current, dict):
            current.pop("operation_id", None)
            current.pop("replica_rank", None)
        return previous == current

    async def create_borrowed_replica(self, spec: dict) -> dict:
        """Create a borrowed runtime while keeping operation metadata idempotent.

        The lock protects rank allocation and duplicate detection only.  Ray
        actor creation is deliberately outside the lock so a slow vLLM
        startup cannot block an unrelated lease request.
        """
        normalized = self._validate_create_spec(spec)
        lease_id = normalized["lease_id"]
        record = None
        async with self.replica_operation_lock:
            existing = self.borrowed_operations.get(lease_id)
            if existing is not None:
                if not self._same_request(existing, normalized):
                    raise ValueError(f"lease_id {lease_id} already has a different create request")
                return self._receipt(existing)
            for record in self.borrowed_operations.values():
                if record.get("operation_id") == normalized["operation_id"]:
                    raise ValueError(f"operation_id {normalized['operation_id']} is already in use")

            replica_rank = self._allocate_replica_rank_locked(normalized["replica_rank"])
            normalized["replica_rank"] = replica_rank
            request_spec = copy.deepcopy(normalized)
            record = {
                "operation_id": normalized["operation_id"],
                "lease_id": lease_id,
                "borrower_task_id": normalized["borrower_task_id"],
                "replica_rank": replica_rank,
                "claim_ids": [claim["claim_id"] for claim in normalized["claims"]],
                "source_lease_ids": normalized["source_lease_ids"],
                "state": "CREATING",
                "cancel_requested": False,
                "request_spec": request_spec,
                "replica": None,
                "worker_handles": [],
                "server_handles": [],
                "created_actor_names": [],
                "result": None,
                "error": None,
            }
            self.borrowed_operations[lease_id] = record


        replica = None
        try:
            replica = MultiTaskvLLMReplica(
                replica_rank=replica_rank,
                config=self.rollout_config,
                model_config=getattr(self, "model_config", None),
                # The parent constructor validates the native world-size layout.
                # Borrowed claims may have a different or fragmented layout, so
                # use a trivially divisible placeholder and replace the topology
                # after claims have been validated in init_from_lease().
                gpus_per_node=1,
                allocation_kind="borrowed",
                lease_id=lease_id,
                source_lease_ids=normalized["source_lease_ids"],
                donor_task_ids=list(dict.fromkeys(claim["donor_task_id"] for claim in normalized["claims"])),
                donor_replica_ranks=list(dict.fromkeys(claim["donor_replica_rank"] for claim in normalized["claims"])),
                owns_resource_pool=False,
                max_colocate_count=normalized["max_colocate_count"],
                claims=normalized["claims"],
                operation_id=normalized["operation_id"],
            )
            runtime = await replica.init_from_lease(normalized)
        except Exception as exc:
            async with self.replica_operation_lock:
                record["state"] = "FAILED"
                record["error"] = {"code": "RUNTIME_CREATION_FAILED", "message": str(exc)}
                cleanup = getattr(replica, "cleanup_result", None) if replica is not None else None
                record["result"] = {
                    "operation_id": record["operation_id"],
                    "lease_id": record["lease_id"],
                    "lease_ids": record["source_lease_ids"],
                    "replica_rank": record["replica_rank"],
                    "state": record["state"],
                    "released": False,
                    "error": record["error"],
                    "cleanup": copy.deepcopy(cleanup),
                }
            return self._receipt(record)

        async with self.replica_operation_lock:
            record["state"] = "RUNTIME_READY"
            record["replica"] = replica
            record["worker_handles"] = list(replica.workers)
            record["server_handles"] = list(replica.servers)
            record["created_actor_names"] = list(replica.created_actor_names)
            record["result"] = {
                "operation_id": record["operation_id"],
                "lease_id": record["lease_id"],
                "lease_ids": record["source_lease_ids"],
                "replica_rank": record["replica_rank"],
                "state": record["state"],
                "released": False,
                "runtime": runtime,
                "error": None,
            }
            return self._receipt(record)

    async def _snapshot_native_claims(self, replica, donor_task_id: str) -> list[dict]:
        """Build physical claims from one initialized native replica for D2 smoke tests.

        This helper is deliberately local to the test hook.  Production claims
        still come from GS; the helper only avoids inventing PG or device IDs in
        a real Ray validation run.
        """
        resource_pool = getattr(replica, "resource_pool", None)
        workers = list(getattr(replica, "workers", []))
        if resource_pool is None or not workers:
            raise RuntimeError("native replica has no initialized resource pool or workers")

        placement_groups = resource_pool.get_placement_groups(device_name=get_device_name())
        local_world_size = int(replica.gpus_per_replica_node)
        if len(workers) != int(replica.world_size) or len(placement_groups) != int(replica.nnodes):
            raise RuntimeError("native replica topology is incomplete for D2 claim snapshot")

        # PlacementGroup is an ID/bundle handle, not a holder of its registered
        # name. Query Ray's metadata once per PG, not once per worker/bundle.
        pg_names = {}
        for placement_group in placement_groups:
            pg_id = placement_group.id.hex()
            pg_info = ray.util.placement_group_table(placement_group)
            pg_name = (pg_info or {}).get("name")
            if not isinstance(pg_name, str) or not pg_name:
                raise RuntimeError(
                    f"native placement group {pg_id} has no registered name in Ray placement_group_table"
                )
            pg_names[pg_id] = pg_name

        def inspect_worker(_worker):
            import os

            import ray
            from verl.utils.device import get_resource_name

            context = ray.get_runtime_context()
            resource_ids = context.get_accelerator_ids().get(get_resource_name(), [])
            if not resource_ids:
                raise RuntimeError("native worker exposes no accelerator id")
            return {
                "node_id": context.get_node_id(),
                "accelerator_id": str(resource_ids[0]),
                "actor_id": str(context.get_actor_id()),
                "pid": os.getpid(),
            }

        worker_infos = await asyncio.gather(*[worker.__ray_call__.remote(inspect_worker) for worker in workers])
        node_rank_by_id: dict[str, int] = {}
        local_rank_by_node: dict[str, int] = {}
        claims = []
        expanded_placement_groups = [
            placement_group for placement_group in placement_groups for _ in range(local_world_size)
        ]
        for rank, (worker_info, placement_group) in enumerate(zip(worker_infos, expanded_placement_groups, strict=True)):
            # Native RayWorkerGroup creates local ranks consecutively inside
            # each PG; the placement group list therefore repeats once per
            # local worker when expanded by local_world_size.
            node_id = worker_info["node_id"]
            node_rank = node_rank_by_id.setdefault(node_id, len(node_rank_by_id))
            local_rank = local_rank_by_node.get(node_id, 0)
            local_rank_by_node[node_id] = local_rank + 1
            pg_id = placement_group.id.hex()
            pg_name = pg_names[pg_id]
            claims.append(
                {
                    "claim_id": f"d2-claim-{replica.replica_rank}-{rank}",
                    "lease_id": f"d2-source-lease-{replica.replica_rank}",
                    "donor_task_id": donor_task_id,
                    "donor_replica_rank": int(replica.replica_rank),
                    "pg_id": pg_id,
                    "pg_name": pg_name,
                    "bundle_index": rank % local_world_size,
                    "node_id": node_id,
                    "gpu_uuid": worker_info["accelerator_id"],
                    "accelerator_id": worker_info["accelerator_id"],
                    "local_gpu_index": local_rank,
                    "node_rank": node_rank,
                    "local_rank": local_rank,
                    "gpu_fraction": 0.5,
                    "cpu_request": 1.0,
                }
            )
        return claims

    @staticmethod
    def _reindex_test_claims(claims: list[dict]) -> list[dict]:
        """Assign borrower-local rank and uniform node/local ranks."""
        node_order: dict[str, int] = {}
        for claim in claims:
            node_order.setdefault(claim["node_id"], len(node_order))
        ordered = sorted(claims, key=lambda item: (node_order[item["node_id"]], item["local_rank"]))
        local_ranks: dict[str, int] = {}
        for rank, claim in enumerate(ordered):
            node_id = claim["node_id"]
            local_rank = local_ranks.get(node_id, 0)
            local_ranks[node_id] = local_rank + 1
            claim["rank"] = rank
            claim["node_rank"] = node_order[node_id]
            claim["local_rank"] = local_rank
            claim["claim_id"] = f"{claim['claim_id']}-borrower-{rank}"
        return ordered

    async def _build_d2_test_spec(self, scenario: str) -> tuple[dict, bool]:
        """Construct one real-placement spec and report whether failure is expected."""
        native_replicas = list(getattr(self, "rollout_replicas", []))
        if not native_replicas:
            raise RuntimeError("D2 test requires at least one initialized native replica")
        donor_task_id = "d2-local-donor-task"
        snapshots = await asyncio.gather(
            *[self._snapshot_native_claims(replica, donor_task_id) for replica in native_replicas]
        )
        source_claims = snapshots[0]
        expected_failure = scenario in {"missing_pg", "duplicate_device", "expired"}

        if scenario == "basic":
            selected = source_claims
        elif scenario == "split":
            if len(source_claims) < 2:
                raise RuntimeError("split scenario needs a native replica with at least two workers")
            selected = source_claims[: len(source_claims) // 2]
        elif scenario == "fragmented":
            if len(source_claims) < 3:
                raise RuntimeError("fragmented scenario needs at least three native workers")
            selected = source_claims[::2]
        elif scenario == "cross_pg":
            if len(snapshots) < 2:
                raise RuntimeError("cross_pg scenario needs at least two native replicas/PGs")
            selected = [snapshots[0][0], snapshots[1][0]]
        elif scenario in {"missing_pg", "duplicate_device", "expired"}:
            if len(source_claims) < 2:
                raise RuntimeError(f"{scenario} scenario needs at least two native workers")
            selected = source_claims[:2]
        else:
            raise ValueError(f"unknown D2 runtime scenario: {scenario}")

        selected = copy.deepcopy(selected)
        if scenario == "missing_pg":
            selected[0]["pg_name"] = f"missing-d2-pg-{time.time_ns()}"
        elif scenario == "duplicate_device":
            selected[1]["node_id"] = selected[0]["node_id"]
            selected[1]["gpu_uuid"] = selected[0]["gpu_uuid"]
            selected[1]["accelerator_id"] = selected[0]["accelerator_id"]
        selected = self._reindex_test_claims(selected)
        world_size = len(selected)
        spec = {
            "operation_id": f"d2-smoke-{scenario}-{time.time_ns()}",
            "lease_id": f"d2-borrower-lease-{scenario}-{time.time_ns()}",
            "lease_ids": list(dict.fromkeys(claim["lease_id"] for claim in selected)),
            "borrower_task_id": "d2-local-borrower-task",
            "borrower_replica_id": f"d2-borrower-{scenario}",
            "claims": selected,
            "world_size": world_size,
            "max_colocate_count": 2,
            "expires_at": time.time() + 600.0,
            "placement_epoch": 1,
            "parallelism": {
                "tensor_model_parallel_size": world_size,
                "data_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
            },
            "creation_timeout_s": 600.0,
        }
        if scenario == "expired":
            spec["expires_at"] = time.time() - 1.0
        return spec, expected_failure

    async def run_d2_runtime_smoke(self, scenario: str, cleanup_after_test: bool = True) -> dict:
        """Run one real-placement D2 create scenario; CE/LB remain untouched."""
        spec, expected_failure = await self._build_d2_test_spec(scenario)
        sleeping_donors = []
        if not expected_failure:
            # The local smoke donor owns the same physical NPU claimed by the
            # borrower. Its standalone server must offload weights first;
            # Ray's fractional GPU accounting does not provide memory isolation.
            sleeping_donors = self._local_native_donors(spec)
            for donor in sleeping_donors:
                await self._test_memory_call(donor, "sleep_for_runtime_test")
            print(f"RUNTIME_TEST_DONORS_SLEEPING {[donor.replica_rank for donor in sleeping_donors]}")
        try:
            receipt = await self.create_borrowed_replica(spec)
        except Exception as exc:
            if not expected_failure:
                # Failed engine startup has no acknowledged device release.
                # Abort this test job; do not wake donors into unknown memory.
                raise
            result = {"scenario": scenario, "status": "EXPECTED_FAILURE", "error": str(exc)}
            print(f"D2_RUNTIME_RESULT {json.dumps(result, sort_keys=True)}")
            return result

        state = receipt.get("state")
        if expected_failure:
            if state not in {"FAILED", "CREATING"}:
                raise RuntimeError(f"D2 scenario {scenario} unexpectedly returned {state}: {receipt}")
            result = {"scenario": scenario, "status": "EXPECTED_FAILURE", "receipt": receipt}
            print(f"D2_RUNTIME_RESULT {json.dumps(result, sort_keys=True, default=str)}")
            return result
        if state != "RUNTIME_READY":
            raise RuntimeError(f"D2 scenario {scenario} did not reach RUNTIME_READY: {receipt}")

        record = self.borrowed_operations.get(spec["lease_id"])
        if record is not None:
            # Keep donor objects private to the manager record.  The receipt
            # remains serializable and does not expose actor handles.
            record["sleeping_donors"] = sleeping_donors

        cleanup = None
        if cleanup_after_test:
            replica = record.get("replica") if record else None
            if replica is not None:
                await self._test_memory_call(replica, "sleep_for_runtime_test")
                cleanup = await replica._cleanup_runtime()
            for donor in sleeping_donors:
                await self._test_memory_call(donor, "wake_for_runtime_test")
            if record is not None:
                record["sleeping_donors"] = []
        result = {
            "scenario": scenario,
            "status": "PASS",
            "receipt": receipt,
            "cleanup": cleanup,
            "sleeping_donor_ranks": [int(donor.replica_rank) for donor in sleeping_donors],
        }
        print(f"D2_RUNTIME_RESULT {json.dumps(result, sort_keys=True, default=str)}")
        return result

    async def reclaim_replica(self, lease_id: str) -> dict:
        """Reserve the reclaim contract without performing lifecycle cleanup."""
        if not isinstance(lease_id, str) or not lease_id:
            raise ValueError("lease_id must be a non-empty string")
        async with self.replica_operation_lock:
            record = self.borrowed_operations.get(lease_id)
            if record is None:
                raise KeyError(f"unknown borrower lease_id: {lease_id}")
            if record["state"] == "DESTROYED":
                return self._receipt(record)
            return {
                "operation_id": record["operation_id"],
                "lease_id": lease_id,
                "lease_ids": record["source_lease_ids"],
                "replica_rank": record["replica_rank"],
                "state": record["state"],
                "released": False,
                "error": {
                    "code": "LIFECYCLE_NOT_IMPLEMENTED",
                    "message": "reclaim_replica is reserved; production runtime cleanup is a later lifecycle stage",
                },
            }

    async def retire_replica_rank(self, replica_rank: int, owner_id: str) -> None:
        """Permanently retire a task-local rank after a future destroy succeeds."""
        if isinstance(replica_rank, bool) or not isinstance(replica_rank, int) or replica_rank < 0:
            raise ValueError("replica_rank must be a non-negative integer")
        if not isinstance(owner_id, str) or not owner_id:
            raise ValueError("owner_id must be a non-empty string")
        async with self.replica_operation_lock:
            matching = [
                record
                for record in self.borrowed_operations.values()
                if record.get("replica_rank") == replica_rank and record.get("borrower_task_id") == owner_id
            ]
            if not matching:
                raise KeyError(f"no replica_rank {replica_rank} owned by {owner_id}")
            self.retired_replica_ranks.add(replica_rank)
