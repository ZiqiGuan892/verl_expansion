"""Experimental Fully Async Rollouter with an extended manager creation point.

The manager/agent-loop initialization below follows verl's Apache-2.0-licensed
FullyAsyncRollouter; only the manager type and its GS handle are added.
"""

import ray

from verl.experimental.fully_async_policy.fully_async_rollouter import (
    FullyAsyncAgentLoopManager,
    FullyAsyncRollouter,
)
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient

from multi_task_scheduler.integration.verl.ray_actor import unwrap_native_actor_class

from .llm_server_manager import MultiTaskLLMServerManager


@ray.remote(num_cpus=10, max_concurrency=100)
class MultiTaskFullyAsyncRollouter(unwrap_native_actor_class(FullyAsyncRollouter)):
    """Real Ray Actor; native generation, queue and training methods stay inherited."""

    def __init__(self, config, tokenizer, processor=None, device_name=None, *, group_scheduler=None):
        self.group_scheduler = group_scheduler
        super().__init__(config, tokenizer, processor=processor, device_name=device_name)

    async def _init_async_rollout_manager(self):
        enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool
        reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None

        assert self.config.actor_rollout_ref.rollout.mode == "async"
        self.async_rollout_mode = True
        self.llm_server_manager = await MultiTaskLLMServerManager.create(
            config=self.config,
            worker_group=self.get_hybrid_worker_group(),
            group_scheduler=self.group_scheduler,
        )
        await self._maybe_run_d2_runtime_smoke()
        self.async_rollout_manager = await FullyAsyncAgentLoopManager.create(
            config=self.config,
            llm_client=self.llm_server_manager.get_client(client_cls=FullyAsyncLLMServerClient),
            reward_loop_worker_handles=reward_loop_worker_handles,
            teacher_client=self.teacher_model_manager.get_client() if self.teacher_model_manager else None,
        )

    async def _maybe_run_d2_runtime_smoke(self) -> None:
        """Run an opt-in D2 creation scenario after native replicas exist.

        The hook is disabled unless the launcher adds
        ``+multitask.d2_runtime_test.enabled=true``.  It deliberately stops at
        ``RUNTIME_READY``: CE registration, parameter bootstrap and LB
        publication belong to later stages and are not called here.
        """
        config_get = getattr(self.config, "get", None)
        multitask_config = (
            config_get("multitask", {})
            if callable(config_get)
            else getattr(self.config, "multitask", {})
        )
        test_config = multitask_config.get("d2_runtime_test", {}) if multitask_config is not None else {}
        if not bool(test_config.get("enabled", False)):
            return
        scenario = str(test_config.get("scenario", "split"))
        cleanup_after_test = bool(test_config.get("cleanup_after_test", True))
        result = await self.llm_server_manager.run_d2_runtime_smoke(
            scenario=scenario,
            cleanup_after_test=cleanup_after_test,
        )
        print(f"[D2 RUNTIME] scenario={scenario} status={result.get('status')}")

    async def run_d3_runtime_smoke(self, scenario: str = "split") -> dict:
        """Create and retain one borrowed runtime for Trainer CE bootstrap."""
        result = await self.llm_server_manager.run_d2_runtime_smoke(
            scenario=scenario,
            cleanup_after_test=False,
        )
        if result.get("status") != "PASS":
            raise RuntimeError(f"D3 runtime preparation did not reach RUNTIME_READY: {result}")
        replica_rank = result["receipt"]["replica_rank"]
        await self.llm_server_manager.register_borrowed_replica_for_ce(replica_rank)
        return {
            "scenario": scenario,
            "replica_rank": replica_rank,
            "receipt": result["receipt"],
            "sleeping_donor_ranks": result.get("sleeping_donor_ranks", []),
        }

    async def create_borrowed_replica(self, spec: dict) -> dict:
        """Thin TaskRunner-facing forwarder to the local runtime manager."""
        return await self.llm_server_manager.create_borrowed_replica(spec)

    async def commit_replica_ready(self, replica_rank: int) -> dict:
        """Thin forwarder for the manager-owned LB READY commit."""
        return await self.llm_server_manager.commit_replica_ready(replica_rank)

    async def prepare_d4_runtime_smoke(self, scenario: str = "split") -> dict:
        """Build a real-placement D4 test spec without creating a runtime.

        The method exists only for the main_ppo smoke entry.  Production
        callers receive an already-authorized spec from GS and never invoke
        this helper or invent placement claims locally.
        """
        # D4 retry/concurrency cases exercise commands on a basic placement;
        # they are not new D2 placement algorithms.
        placement_scenario = "basic" if scenario in {"idempotent", "concurrent_idempotent"} else scenario
        spec, expected_failure = await self.llm_server_manager._build_d2_test_spec(placement_scenario)
        sleeping = {"state": "NOT_REQUIRED", "replica_ranks": []}
        if not expected_failure:
            sleeping = await self.llm_server_manager.sleep_d4_test_donors(spec)
        return {"spec": spec, "expected_failure": expected_failure, "sleeping": sleeping}

    async def probe_replica_ready(self, replica_rank: int) -> dict:
        """Verify the borrowed primary endpoint is present in the LB route table."""
        return await self.llm_server_manager.probe_replica_ready(replica_rank)

    async def test_operation_snapshot(self, lease_id: str) -> dict:
        """Return the test-only idempotency projection without exposing handles."""
        return await self.llm_server_manager.test_operation_snapshot(lease_id)

    async def run_d4_shared_bundle_smoke(self) -> dict:
        """Run the opt-in same-bundle placement fixture owned by the manager."""
        return await self.llm_server_manager.run_d4_shared_bundle_smoke()

    async def cleanup_d4_runtime(self, replica_rank: int) -> dict:
        """Run test-only route removal and actor cleanup after D4 smoke."""
        return await self.llm_server_manager.cleanup_d4_runtime(replica_rank)

    async def get_borrowed_replica_for_ce(self, replica_rank: int):
        """Return one manager-owned borrowed replica to the Trainer actor."""
        return await self.llm_server_manager.get_replica_for_ce(replica_rank)

    async def cleanup_d3_runtime(self, replica_rank: int, global_steps: int | None = None) -> dict:
        """Release only the actors created by the D3 smoke scenario."""
        # TODO(lifecycle): production reclaim/destroy must be coordinated by
        # the lifecycle owner, not routed through this D3-only cleanup hook.
        return await self.llm_server_manager.cleanup_d3_runtime(replica_rank, global_steps=global_steps)

    async def mark_replica_serving_version(self, replica_rank: int, version: int) -> dict:
        """Project the CE-confirmed version onto the manager-owned replica."""
        return await self.llm_server_manager.mark_replica_serving_version(replica_rank, version)
