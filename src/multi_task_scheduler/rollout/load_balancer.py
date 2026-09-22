"""Native routing subclass with a GS reference and no scheduling side effects."""

from verl.workers.rollout.router import DEFAULT_ROUTING_CACHE_SIZE, GlobalRequestLoadBalancer


class MultiTaskGlobalRequestLoadBalancer(GlobalRequestLoadBalancer):
    """Manager wraps this ordinary class in Ray; all routing methods stay inherited."""

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
