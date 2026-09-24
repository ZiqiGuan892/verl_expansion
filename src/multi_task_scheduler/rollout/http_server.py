"""Native vLLM server extension with no new generation behavior."""

from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer


class MultiTaskvLLMHttpServer(vLLMHttpServer):
    """Replica wraps this class in Ray; every native method remains inherited."""

    def _require_test_sleep_mode(self) -> None:
        if not self.config.enable_sleep_mode or not self.config.free_cache_engine:
            raise RuntimeError("runtime memory test requires enable_sleep_mode=true and free_cache_engine=true")

    async def sleep_for_runtime_test(self) -> None:
        """Offload an idle test engine; native standalone sleep() is a no-op.

        Level 1 copies weights to CPU and discards KV cache, releasing both
        device allocations while preserving weights for wake-up. This is only
        called by startup tests before generation, not a production drain API.
        """
        if self.node_rank != 0:
            return
        self._require_test_sleep_mode()
        await self.engine.sleep(level=1)

    async def wake_for_runtime_test(self) -> None:
        """Restore the donor's saved weights and cache before native training."""
        if self.node_rank != 0:
            return
        self._require_test_sleep_mode()
        await self.engine.wake_up(tags=["weights", "kv_cache"])
        await self.engine.reset_prefix_cache(reset_connector=True)
        # TODO(lifecycle): production wake must target-sync the latest actor
        # parameters before this server accepts traffic. Engine wake only
        # restores the frozen sleep snapshot; it does not perform CE sync.
