"""Checkpoint manager extensions for borrowed-replica membership and bootstrap."""

import asyncio
import hashlib
import json

import ray

from verl.checkpoint_engine.base import (
    CheckpointEngineManager,
    _worker_cls,
)
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.utils.device import get_device_name


class MultiTaskCheckpointEngineManager(CheckpointEngineManager):
    """Extend native CE synchronization with target-only replica bootstrap.

    The object remains trainer-local.  It stores rollout replica projections and
    worker handles, but never owns the global scheduler or the load balancer.
    Native ``update_weights`` is reused for the normal effective set; borrowed
    bootstrap uses the same backend primitives with a temporary target group.
    """

    def __init__(self, config, actor_wg, replicas):
        super().__init__(config=config, actor_wg=actor_wg, replicas=replicas)
        self.sync_gate = asyncio.Lock()
        self.sync_state = "IDLE"
        self.inflight_replicas = []
        self.last_synced_versions: dict[int, int] = {}
        self.pending_bootstrap: dict[int, int | None] = {}
        # Full per-parameter receiver verification is intentionally opt-in:
        # hashing every model tensor adds measurable transfer-time overhead.
        # D3/D4 acceptance scripts enable it when they need strict evidence.
        self.parameter_validation_enabled = False
        # Stricter actor-source to CE-receiver comparison.  The configured
        # checkpoint backend must expose get_source_manifest() on actor rank 0.
        self.source_validation_enabled = False
        # Replica ranks temporarily excluded from the effective CE set while
        # their physical slots are leased to a borrowed replica.  The donor
        # server is slept by the rollout manager; this set prevents its CE
        # worker from joining a collective on the same device as the borrower.
        self.suspended_replica_ranks: set[int] = set()

    @staticmethod
    def _replica_rank(replica) -> int:
        rank = getattr(replica, "replica_rank", None)
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
            raise ValueError("replica must expose a non-negative integer replica_rank")
        return rank

    @staticmethod
    def _handle_key(handle):
        actor_id = getattr(handle, "_actor_id", None)
        if actor_id is None:
            return str(handle)
        hex_method = getattr(actor_id, "hex", None)
        return hex_method() if callable(hex_method) else str(actor_id)

    @classmethod
    def _same_workers(cls, left, right) -> bool:
        return [cls._handle_key(item) for item in left] == [cls._handle_key(item) for item in right]

    def _find_replica_unlocked(self, replica_rank: int):
        for replica in self.replicas:
            if getattr(replica, "replica_rank", None) == replica_rank:
                return replica
        return None

    def _effective_replicas_unlocked(self):
        return [
            replica
            for replica in self.replicas
            if getattr(replica, "replica_rank", None) not in self.pending_bootstrap
            and getattr(replica, "replica_rank", None) not in self.suspended_replica_ranks
        ]

    @staticmethod
    def _normalize_replica_ranks(replica_ranks) -> set[int]:
        ranks = set(replica_ranks)
        if any(isinstance(rank, bool) or not isinstance(rank, int) or rank < 0 for rank in ranks):
            raise ValueError("replica_ranks must contain non-negative integers")
        return ranks

    async def suspend_replicas_for_sync(self, replica_ranks) -> dict:
        """Exclude donor ranks from the next effective CE synchronization.

        This is used by the D3 same-slot smoke only after the donor servers
        have entered Engine sleep.  Leaving donor CE workers in the HCCL
        topology would create duplicate ranks on the borrowed physical
        devices and causes HCCL ``parameter error`` during communicator init.

        The method changes only the manager's effective-replica projection.
        It deliberately does not drain requests, update LB routing, sleep a
        vLLM server, or finalize an already-built communication domain.
        TODO(lifecycle): the production owner must perform those operations
        before the donor is reused by a borrowed replica, then call this
        method while holding the same lifecycle/synchronization boundary.
        """
        ranks = self._normalize_replica_ranks(replica_ranks)
        async with self.sync_gate:
            known = {getattr(replica, "replica_rank", None) for replica in self.replicas}
            missing = sorted(ranks - known)
            if missing:
                raise KeyError(f"unknown replica ranks: {missing}")
            self.suspended_replica_ranks.update(ranks)
            return {"state": "SUSPENDED", "replica_ranks": sorted(ranks)}

    async def resume_replicas_for_sync(self, replica_ranks) -> dict:
        """Allow previously suspended donor ranks into future CE snapshots.

        This is only the final CE membership step.  The caller must first
        wake the donor, synchronize it to the current actor version with a
        target-only CE transaction, and finalize that temporary communication
        domain.  This method does not perform any of those operations.
        TODO(lifecycle): enforce a production wake/target-sync receipt before
        removing a rank from ``suspended_replica_ranks``.  The D3 smoke keeps
        the weaker behavior so its test fixture remains runnable.
        """
        ranks = self._normalize_replica_ranks(replica_ranks)
        async with self.sync_gate:
            self.suspended_replica_ranks.difference_update(ranks)
            return {"state": "RESUMED", "replica_ranks": sorted(ranks)}

    async def register_replica(self, replica) -> dict:
        """Add a RUNTIME_READY replica to the CE projection as pending.

        Registration is idempotent for the same rank and worker handles.  The
        replica is excluded from ordinary full-set synchronization until its
        target-only bootstrap succeeds.
        """
        rank = self._replica_rank(replica)
        workers = list(getattr(replica, "workers", []))
        if len(workers) != int(getattr(replica, "world_size", -1)) or not workers:
            raise ValueError("replica worker count must equal replica world_size")
        async with self.sync_gate:
            current = self._find_replica_unlocked(rank)
            if current is not None:
                if not self._same_workers(getattr(current, "workers", []), workers):
                    raise ValueError(f"replica_rank {rank} is already registered with different workers")
                return {
                    "replica_rank": rank,
                    "state": "ALREADY_REGISTERED",
                    "pending": rank in self.pending_bootstrap,
                }
            self.replicas.append(replica)
            self.pending_bootstrap[rank] = None
            return {"replica_rank": rank, "state": "REGISTERED", "pending": True}

    async def unregister_replica(self, replica_or_rank) -> dict:
        """Remove a replica from future CE snapshots and version tracking."""
        if isinstance(replica_or_rank, int) and not isinstance(replica_or_rank, bool):
            rank = replica_or_rank
            if rank < 0:
                raise ValueError("replica_rank must be a non-negative integer")
        else:
            rank = self._replica_rank(replica_or_rank)
        async with self.sync_gate:
            replica = self._find_replica_unlocked(rank)
            if replica is None:
                return {"replica_rank": rank, "state": "NOT_REGISTERED"}
            self.replicas = [item for item in self.replicas if getattr(item, "replica_rank", None) != rank]
            self.pending_bootstrap.pop(rank, None)
            self.suspended_replica_ranks.discard(rank)
            self.last_synced_versions.pop(rank, None)
            if getattr(replica, "serving_version", None) is not None:
                replica.serving_version = None
            return {"replica_rank": rank, "state": "UNREGISTERED"}

    async def update_weights(self, global_steps: int = None):
        """Run native full-set sync while serializing membership changes."""
        async with self.sync_gate:
            if self.sync_state == "BLOCKED":
                raise RuntimeError("checkpoint manager is BLOCKED after a previous sync failure")
            self.sync_state = "SYNCING"
            self.inflight_replicas = list(self._effective_replicas_unlocked())
            previous = self.replicas
            self.replicas = list(self.inflight_replicas)
            try:
                result = await super().update_weights(global_steps=global_steps)
                version = int(global_steps) if global_steps is not None else None
                if self.parameter_validation_enabled or self.source_validation_enabled:
                    source_manifest = await self._get_source_manifest() if self.source_validation_enabled else None
                    validation = await self.validate_parameter_sync(
                        replicas=self.inflight_replicas,
                        expected_version=version,
                        source_manifest=source_manifest,
                    )
                    if isinstance(result, dict):
                        result["parameter_validation"] = validation
                    print(f"CE_PARAMETER_VALIDATION {json.dumps(validation, sort_keys=True, default=str)}")
                if version is not None:
                    for replica in self.inflight_replicas:
                        rank = self._replica_rank(replica)
                        self.last_synced_versions[rank] = version
                        replica.serving_version = version
                self.sync_state = "IDLE"
                return result
            except Exception:
                self.sync_state = "BLOCKED"
                raise
            finally:
                self.replicas = previous
                if self.sync_state == "IDLE":
                    self.inflight_replicas = []

    async def bootstrap_replica(self, replica, snapshot_version: int) -> dict:
        """Synchronize one pending replica with the frozen trainer snapshot.

        The temporary worker group contains only ``replica.workers``.  The
        actor group and target group use the native backend topology builder,
        then finalize the group before the replica is made effective.
        """
        rank = self._replica_rank(replica)
        if isinstance(snapshot_version, bool) or not isinstance(snapshot_version, int) or snapshot_version < 0:
            raise ValueError("snapshot_version must be a non-negative integer")
        async with self.sync_gate:
            registered = self._find_replica_unlocked(rank)
            if registered is None or not self._same_workers(getattr(registered, "workers", []), replica.workers):
                raise ValueError(f"replica_rank {rank} is not registered in this checkpoint manager")
            if rank not in self.pending_bootstrap:
                if self.last_synced_versions.get(rank) == snapshot_version:
                    return {"replica_rank": rank, "state": "WEIGHTS_READY", "version": snapshot_version}
                raise ValueError(f"replica_rank {rank} is not pending bootstrap")
            if self.backend == "naive":
                raise NotImplementedError("target-only bootstrap requires a distributed checkpoint backend")

            self.sync_state = "SYNCING"
            self.inflight_replicas = [registered]
            target_group = None
            group_initialized = False
            kv_released = False
            generation_aborted = False
            finalized = False
            kv_resumed = False
            generation_resumed = False
            try:
                target_group = RayWorkerGroup.from_detached(
                    name_prefix=f"bootstrap_{rank}_{snapshot_version}",
                    worker_handles=list(registered.workers),
                    ray_cls_with_init=RayClassWithInitArgs(cls=_worker_cls),
                    device_name=get_device_name(),
                )
                await registered.abort_all_requests()
                generation_aborted = True
                await registered.release_kv_cache()
                kv_released = True
                self.build_process_group(target_group)
                group_initialized = True
                ray.get(
                    self.actor_wg.update_weights(global_steps=snapshot_version, mode=self.backend)
                    + target_group.update_weights(global_steps=snapshot_version)
                )
                ray.get(
                    self.actor_wg.execute_checkpoint_engine(["finalize"] * self.actor_wg.world_size)
                    + target_group.execute_checkpoint_engine(["finalize"] * target_group.world_size)
                )
                finalized = True
                await registered.resume_kv_cache()
                kv_resumed = True
                await registered.resume_generation()
                generation_resumed = True
                if self.parameter_validation_enabled or self.source_validation_enabled:
                    source_manifest = await self._get_source_manifest() if self.source_validation_enabled else None
                    validation = await self.validate_parameter_sync(
                        replicas=[registered],
                        expected_version=snapshot_version,
                        source_manifest=source_manifest,
                    )
                    print(f"CE_PARAMETER_VALIDATION {json.dumps(validation, sort_keys=True, default=str)}")
                self.last_synced_versions[rank] = snapshot_version
                self.pending_bootstrap.pop(rank, None)
                registered.serving_version = snapshot_version
                self.sync_state = "IDLE"
                result = {
                    "replica_rank": rank,
                    "state": "WEIGHTS_READY",
                    "version": snapshot_version,
                    "communication": "target_only",
                    "finalized": True,
                }
                if self.parameter_validation_enabled or self.source_validation_enabled:
                    result["parameter_validation"] = validation
                return result
            except Exception:
                self.sync_state = "BLOCKED"
                # Keep pending_bootstrap so the failed replica cannot enter the
                # normal effective set or be reported as serving-ready.
                raise
            finally:
                if group_initialized and not finalized:
                    try:
                        ray.get(
                            self.actor_wg.execute_checkpoint_engine(["finalize"] * self.actor_wg.world_size)
                            + target_group.execute_checkpoint_engine(["finalize"] * target_group.world_size)
                        )
                    except Exception:
                        # Preserve the original bootstrap exception; the manager
                        # remains BLOCKED until an operator handles the failed
                        # communication transaction.
                        pass
                if kv_released and not kv_resumed:
                    try:
                        await registered.resume_kv_cache()
                    except Exception:
                        pass
                if generation_aborted and not generation_resumed:
                    try:
                        await registered.resume_generation()
                    except Exception:
                        pass
                if self.sync_state == "IDLE":
                    self.inflight_replicas = []

    @staticmethod
    def _manifest_digest(manifest: dict) -> str:
        entries = manifest.get("parameters", [])
        canonical = [
            {
                "name": item.get("name"),
                "shape": list(item.get("shape", [])),
                "dtype": item.get("dtype"),
                "numel": int(item.get("numel", 0)),
                "sha256": item.get("sha256"),
            }
            for item in entries
        ]
        canonical.sort(key=lambda item: item["name"] or "")
        encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    async def _get_source_manifest(self) -> dict:
        """Read the canonical source manifest from actor checkpoint rank 0."""
        refs = self.actor_wg.execute_checkpoint_engine(["get_source_manifest"] * self.actor_wg.world_size)
        manifests = ray.get(refs)
        for manifest in manifests:
            if isinstance(manifest, dict) and manifest.get("complete", False):
                return manifest
        raise RuntimeError(f"actor source manifest is unavailable or incomplete: {manifests}")

    @staticmethod
    def _manifest_mismatches(expected: dict, actual: dict, limit: int = 8) -> list[dict]:
        expected_entries = {item.get("name"): item for item in expected.get("parameters", [])}
        actual_entries = {item.get("name"): item for item in actual.get("parameters", [])}
        mismatches = []
        for name in sorted(set(expected_entries) | set(actual_entries)):
            source = expected_entries.get(name)
            received = actual_entries.get(name)
            if source is None or received is None:
                mismatches.append({"name": name, "source": source, "received": received})
                continue
            fields = ("shape", "dtype", "numel", "sha256")
            differences = {
                field: {"source": source.get(field), "received": received.get(field)}
                for field in fields
                if source.get(field) != received.get(field)
            }
            if differences:
                mismatches.append({"name": name, "differences": differences})
            if len(mismatches) >= limit:
                break
        return mismatches

    async def validate_parameter_sync(
        self, replicas, expected_version: int | None = None, source_manifest: dict | None = None
    ) -> dict:
        """Verify every received parameter on every CE Worker.

        The receiver worker records a SHA-256 fingerprint while streaming each
        named tensor into ``ServerAdapter``.  This method compares the complete
        name/shape/dtype/value manifest across all workers and checks the
        frozen sync version.  When ``source_manifest`` is provided, every
        receiver is also compared against the actor source field by field.
        It returns metadata only; tensor payloads never leave worker processes.
        """
        workers = [worker for replica in replicas for worker in getattr(replica, "workers", [])]
        if not workers:
            raise RuntimeError("parameter validation requires at least one CE Worker")
        manifests = ray.get([worker.get_parameter_manifest.remote() for worker in workers])
        if not manifests or any(not manifest.get("complete", False) for manifest in manifests):
            raise RuntimeError(f"CE parameter manifest is incomplete: {manifests}")
        if expected_version is not None:
            mismatched_versions = [
                manifest.get("global_steps") for manifest in manifests if manifest.get("global_steps") != expected_version
            ]
            if mismatched_versions:
                raise RuntimeError(
                    f"CE parameter version mismatch: expected={expected_version}, received={mismatched_versions}"
                )
        if source_manifest is not None:
            if not source_manifest.get("complete", False):
                raise RuntimeError(f"actor source manifest is incomplete: {source_manifest}")
            if expected_version is not None and source_manifest.get("global_steps") != expected_version:
                raise RuntimeError(
                    f"actor source parameter version mismatch: expected={expected_version}, "
                    f"received={source_manifest.get('global_steps')}"
                )
            for worker_index, manifest in enumerate(manifests):
                mismatches = self._manifest_mismatches(source_manifest, manifest)
                if mismatches:
                    raise RuntimeError(
                        f"CE worker {worker_index} differs from actor source manifest: {mismatches}"
                    )
        digests = [self._manifest_digest(manifest) for manifest in manifests]
        if len(set(digests)) != 1:
            raise RuntimeError(f"CE workers received different parameter manifests: {digests}")
        first = manifests[0]
        result = {
            "state": "PARAMETERS_VALIDATED",
            "version": first.get("global_steps"),
            "worker_count": len(manifests),
            "parameter_count": first.get("parameter_count", 0),
            "total_numel": first.get("total_numel", 0),
            "manifest_digest": digests[0],
        }
        if source_manifest is not None:
            result.update(
                {
                    "source_state": "SOURCE_TO_RECEIVER_VALIDATED",
                    "source_manifest_digest": self._manifest_digest(source_manifest),
                }
            )
        return result
