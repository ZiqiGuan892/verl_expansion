"""Native replica selection plus D1 borrowed-replica contract state."""

import ray

from verl.single_controller.ray import RayClassWithInitArgs
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica

from multi_task_scheduler.checkpoint.checkpoint_engine_worker import MultiTaskCheckpointEngineWorker

from .http_server import MultiTaskvLLMHttpServer


class MultiTaskvLLMReplica(vLLMReplica):
    """Native-compatible replica with metadata-only D1 lifecycle boundaries.

    The class intentionally does not create a borrowed Worker or server yet.
    Its D2 ``init_from_lease`` implementation will be added only after the D1
    contract and idempotency tests pass.
    """

    def __init__(self, *args, **kwargs):
        self.allocation_kind = kwargs.pop("allocation_kind", "native")
        self.lease_id = kwargs.pop("lease_id", None)
        self.source_lease_ids = list(kwargs.pop("source_lease_ids", []))
        self.donor_task_ids = list(kwargs.pop("donor_task_ids", []))
        self.donor_replica_ranks = list(kwargs.pop("donor_replica_ranks", []))
        self.runtime_state = kwargs.pop("runtime_state", "CREATING")
        self.owns_resource_pool = bool(kwargs.pop("owns_resource_pool", self.allocation_kind == "native"))
        self.max_colocate_count = kwargs.pop("max_colocate_count", None)
        self.claims = list(kwargs.pop("claims", []))
        self.serving_version = kwargs.pop("serving_version", None)
        if self.allocation_kind not in {"native", "borrowed"}:
            raise ValueError("allocation_kind must be 'native' or 'borrowed'")
        if self.allocation_kind == "native" and self.lease_id is not None:
            raise ValueError("native replica cannot have a lease_id")
        if self.allocation_kind == "borrowed" and self.owns_resource_pool:
            raise ValueError("borrowed replica cannot own a resource pool")
        super().__init__(*args, **kwargs)
        self.server_class = ray.remote(MultiTaskvLLMHttpServer)

    def get_ray_class_with_init_args(self) -> RayClassWithInitArgs:
        return RayClassWithInitArgs(
            cls=ray.remote(MultiTaskCheckpointEngineWorker),
            rollout_config=self.config,
            model_config=self.model_config,
            replica_rank=self.replica_rank,
        )

    async def init_from_lease(self, spec: dict) -> None:
        """D1 boundary for borrowed creation; D2 supplies the runtime body."""
        self.runtime_state = "FAILED"
        raise NotImplementedError("borrowed runtime creation is deferred to D2")

    @staticmethod
    def _lifecycle_receipt(code: str, message: str, *, lease_id: str | None, state: str) -> dict:
        return {
            "operation_id": None,
            "lease_id": lease_id,
            "replica_rank": None,
            "state": state,
            "released": False,
            "error": {"code": code, "message": message},
        }

    async def destroy(self) -> dict:
        """Reserve the destroy contract without closing runtime resources in D1."""
        return self._lifecycle_receipt(
            "LIFECYCLE_NOT_IMPLEMENTED",
            "D1 reserves destroy; server, worker and communication cleanup is deferred",
            lease_id=self.lease_id,
            state=self.runtime_state,
        )

    async def reclaim(self, lease_id: str) -> dict:
        """Reserve borrowed reclaim and reject wrong-owner leases explicitly."""
        if self.allocation_kind != "borrowed":
            return self._lifecycle_receipt(
                "INVALID_ALLOCATION_KIND",
                "native replica cannot be reclaimed by borrower lease",
                lease_id=lease_id,
                state=self.runtime_state,
            )
        if lease_id != self.lease_id:
            return self._lifecycle_receipt(
                "LEASE_MISMATCH",
                "lease_id does not own this borrowed replica",
                lease_id=lease_id,
                state=self.runtime_state,
            )
        return self._lifecycle_receipt(
            "LIFECYCLE_NOT_IMPLEMENTED",
            "D1 reserves reclaim; runtime cleanup is deferred",
            lease_id=lease_id,
            state=self.runtime_state,
        )
