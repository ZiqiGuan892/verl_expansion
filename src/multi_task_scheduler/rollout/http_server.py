"""Native vLLM server extension with opt-in test observations."""

import copy

from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer


class MultiTaskvLLMHttpServer(vLLMHttpServer):
    """Replica wraps this class in Ray; generation delegates to native vLLM."""

    async def set_test_generation_audit(self, enabled: bool = True, reset: bool = True) -> dict:
        """Enable idle-window test counters without changing generated output."""
        if not isinstance(enabled, bool) or not isinstance(reset, bool):
            raise TypeError("audit enabled and reset must be booleans")
        current = getattr(self, "_test_generation_audit", None)
        if current is not None and current["inflight"]:
            raise RuntimeError("generation audit can only be changed while idle")
        if reset or current is None:
            self._test_generation_audit = {
                "enabled": enabled,
                "started_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "nonempty_calls": 0,
                "token_count": 0,
                "inflight": 0,
                "peak_inflight": 0,
                "calls_by_version": {},
                "nonempty_by_version": {},
            }
        else:
            current["enabled"] = enabled
        return self.get_test_generation_audit()

    def get_test_generation_audit(self) -> dict:
        """Return a serializable snapshot; counters are disabled by default."""
        audit = getattr(self, "_test_generation_audit", None)
        if audit is None:
            return {"enabled": False}
        return copy.deepcopy(audit)

    async def generate(self, *args, **kwargs):
        audit = getattr(self, "_test_generation_audit", None)
        if audit is None or not audit["enabled"]:
            return await super().generate(*args, **kwargs)
        audit["started_calls"] += 1
        audit["inflight"] += 1
        audit["peak_inflight"] = max(audit["peak_inflight"], audit["inflight"])
        try:
            output = await super().generate(*args, **kwargs)
        except BaseException:
            audit["failed_calls"] += 1
            raise
        else:
            audit["successful_calls"] += 1
            version = str(output.extra_fields.get("global_steps"))
            audit["calls_by_version"][version] = audit["calls_by_version"].get(version, 0) + 1
            if output.token_ids and output.stop_reason not in {"aborted", "abort"}:
                audit["nonempty_calls"] += 1
                audit["token_count"] += len(output.token_ids)
                audit["nonempty_by_version"][version] = audit["nonempty_by_version"].get(version, 0) + 1
            return output
        finally:
            audit["inflight"] -= 1

    def _require_test_sleep_mode(self) -> None:
        if not self.config.enable_sleep_mode or not self.config.free_cache_engine:
            raise RuntimeError("runtime memory test requires enable_sleep_mode=true and free_cache_engine=true")

    async def sleep_for_runtime_test(self) -> None:
        """Offload an idle test engine; native standalone sleep() is a no-op.

        Level 1 copies weights to CPU and discards KV cache, releasing both
        device allocations while preserving weights for wake-up. This is only
        called by startup tests before generation, not a production drain API.

        TODO(lifecycle): the production sleep hook is owned by the lifecycle
        implementation. It must drain/abort requests and coordinate LB and CE
        membership before invoking an engine sleep operation.
        """
        if self.node_rank != 0:
            return
        self._require_test_sleep_mode()
        await self.engine.sleep(level=1)

    async def wake_for_runtime_test(self) -> None:
        """Restore the donor's saved weights and cache for a smoke test.

        This helper is intentionally not the production wake operation.
        """
        if self.node_rank != 0:
            return
        self._require_test_sleep_mode()
        await self.engine.wake_up(tags=["weights", "kv_cache"])
        await self.engine.reset_prefix_cache(reset_connector=True)
        # TODO(lifecycle): production wake must target-sync the latest actor
        # parameters before this server accepts traffic. Engine wake only
        # restores the frozen sleep snapshot; it does not perform CE sync.
