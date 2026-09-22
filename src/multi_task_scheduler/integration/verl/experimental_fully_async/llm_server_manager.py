"""Select rollout subclasses and own D1 borrowed-replica contracts.

The D1 implementation deliberately stops before creating a Ray actor.  It
normalises and validates the metadata contract, allocates a stable task-local
replica rank, and records an explicit non-success receipt.  Runtime creation
from claims is D2 and must not be hidden behind this module yet.
"""

import asyncio
import copy
import math
import time

import ray

from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncLLMServerManager
from verl.workers.rollout.router import DEFAULT_ROUTING_CACHE_SIZE

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
        return copy.deepcopy(record["result"])

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
        """Register a D1 create request without starting a runtime.

        D2 will replace the explicit failure below with the claim-based runtime
        creation path.  Keeping the failure explicit prevents callers from
        treating a metadata-only record as ``RUNTIME_READY``.
        """
        normalized = self._validate_create_spec(spec)
        lease_id = normalized["lease_id"]
        async with self.replica_operation_lock:
            existing = self.borrowed_operations.get(lease_id)
            if existing is not None:
                if not self._same_request(existing, normalized):
                    raise ValueError(f"lease_id {lease_id} already has a different create request")
                return self._receipt(existing)
            for record in self.borrowed_operations.values():
                if record.get("operation_id") == normalized["operation_id"]:
                    raise ValueError(f"operation_id {normalized['operation_id']} is already in use")

            request_spec = copy.deepcopy(normalized)
            replica_rank = self._allocate_replica_rank_locked(normalized["replica_rank"])
            normalized["replica_rank"] = replica_rank
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

            # No runtime creation is allowed in D1.  The rank and frozen
            # request remain recorded so a retry cannot allocate another rank.
            record["state"] = "FAILED"
            record["error"] = {
                "code": "RUNTIME_CREATION_NOT_IMPLEMENTED",
                "message": "D1 records the contract only; borrowed runtime creation is a D2 operation",
            }
            record["result"] = {
                "operation_id": record["operation_id"],
                "lease_id": record["lease_id"],
                "lease_ids": record["source_lease_ids"],
                "replica_rank": record["replica_rank"],
                "state": record["state"],
                "released": False,
                "error": record["error"],
            }
            return self._receipt(record)

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
                    "message": "D1 reserves reclaim_replica; runtime cleanup is a later lifecycle stage",
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
