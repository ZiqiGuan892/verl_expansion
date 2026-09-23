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
