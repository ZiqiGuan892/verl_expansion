"""Native routing subclass with a GS reference and no scheduling side effects."""

from verl.workers.rollout.router import DEFAULT_ROUTING_CACHE_SIZE, GlobalRequestLoadBalancer


class MultiTaskGlobalRequestLoadBalancer(GlobalRequestLoadBalancer):
    """Native router plus an idempotent READY publication boundary.

    The actor remains the only owner of its routing tables.  Lifecycle code
    calls ``commit_ready`` only after CE bootstrap and health checks have
    succeeded; drain/remove and request migration remain out of D4.
    """

    def __init__(
        self,
        servers,
        max_cache_size=DEFAULT_ROUTING_CACHE_SIZE,
        full_determinism=False,
        *,
        group_scheduler=None,
    ):
        self.group_scheduler = group_scheduler
        super().__init__(servers, max_cache_size=max_cache_size, full_determinism=full_determinism)

    @staticmethod
    def _same_handle(left, right) -> bool:
        left_id = getattr(left, "_actor_id", None)
        right_id = getattr(right, "_actor_id", None)
        if left_id is not None or right_id is not None:
            return left_id == right_id
        return left == right

    def commit_ready(self, servers: dict[str, object]) -> dict:
        """Atomically publish validated primary server handles as READY.

        Existing identical ``server_id``/handle pairs are idempotent and keep
        their in-flight counters.  A different handle under an existing ID is
        rejected so an old runtime cannot be silently replaced by a retry.
        """
        if not isinstance(servers, dict) or not servers:
            raise ValueError("servers must be a non-empty mapping")
        for server_id, handle in servers.items():
            if not isinstance(server_id, str) or not server_id:
                raise ValueError("server_id must be a non-empty string")
            if handle is None:
                raise ValueError(f"server {server_id!r} has no actor handle")
            if server_id in self._servers and not self._same_handle(self._servers[server_id], handle):
                raise ValueError(f"server_id {server_id!r} is already owned by another runtime")

        added = []
        for server_id, handle in servers.items():
            if server_id in self._servers:
                continue
            self._servers[server_id] = handle
            self._inflight_requests[server_id] = 0
            added.append(server_id)
        return {"state": "READY", "server_ids": sorted(servers), "added": added}
