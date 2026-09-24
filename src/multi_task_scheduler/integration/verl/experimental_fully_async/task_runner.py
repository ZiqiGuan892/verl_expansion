# Copyright 2025 Meituan Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Replace creation targets while inheriting the native initialization and fit loop."""

import logging
import json
import threading

import ray

from verl.experimental.fully_async_policy.fully_async_main import FullyAsyncTaskRunner
from verl.experimental.separation.utils import create_resource_pool_manager
from verl.trainer.ppo.utils import Role

from multi_task_scheduler.integration.verl.ray_actor import unwrap_native_actor_class
from multi_task_scheduler.scheduler.discovery import get_or_create_group_scheduler

from .rollouter import MultiTaskFullyAsyncRollouter
from .trainer import MultiTaskFullyAsyncTrainer

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=1, max_concurrency=8)
class MultiTaskFullyAsyncTaskRunner(unwrap_native_actor_class(FullyAsyncTaskRunner)):
    """Own GS/Trainer/Rollouter handles, but no rollout or CE Manager objects."""

    def __init__(self):
        super().__init__()
        self.group_scheduler = None
        self._replica_operation_lock = threading.Lock()

    def run(self, config):
        """Attach this Actor to GS, then execute verl's original run method."""
        self.group_scheduler = get_or_create_group_scheduler()
        context = ray.get_runtime_context()
        task_id = context.get_actor_id()
        try:
            ray.get(self.group_scheduler.attach_task.remote(task_id, context.current_actor), timeout=120)
            result = super().run(config)
            self._maybe_run_d4_runtime_smoke(config)
            return result
        finally:
            try:
                ray.get(self.group_scheduler.detach_task.remote(task_id), timeout=30)
            except Exception:
                # Cleanup must not replace the original initialization/training error.
                logger.warning("Could not detach TaskRunner %s from GroupScheduler", task_id, exc_info=True)

    def execute_replica_operation(self, operation: str, request: dict | None = None) -> dict:
        """Execute one task-local create command and return metadata only.

        GS is expected to call this TaskRunner method.  The method keeps the
        orchestration boundary in the task actor: Rollouter creates runtime,
        Trainer registers/bootstrap synchronizes it, and Rollouter commits LB
        READY.  It never returns Ray handles or a PG object.
        """
        request = {} if request is None else request
        if not isinstance(request, dict):
            raise TypeError("replica operation request must be a mapping")
        if operation != "create":
            return {
                "operation_id": request.get("operation_id"),
                "lease_id": request.get("lease_id"),
                "state": "LIFECYCLE_NOT_IMPLEMENTED",
                "released": False,
                "error": {"code": "LIFECYCLE_NOT_IMPLEMENTED", "message": f"operation={operation!r}"},
            }
        with self._replica_operation_lock:
            rollouter = self.components.get("rollouter")
            trainer = self.components.get("trainer")
            if rollouter is None or trainer is None:
                return {
                    "operation_id": request.get("operation_id"),
                    "lease_id": request.get("lease_id"),
                    "state": "NOT_READY",
                    "released": False,
                    "error": {"code": "COMPONENTS_NOT_READY", "message": "TaskRunner components are not initialized"},
                }
            runtime = ray.get(rollouter.create_borrowed_replica.remote(request))
            if runtime.get("state") != "RUNTIME_READY":
                return runtime
            replica_rank = int(runtime["replica_rank"])
            registration = ray.get(trainer.register_replica.remote(replica_rank))
            bootstrap = ray.get(trainer.bootstrap_replica.remote(replica_rank))
            ready = ray.get(rollouter.commit_replica_ready.remote(replica_rank))
            return {
                "operation_id": runtime.get("operation_id"),
                "lease_id": runtime.get("lease_id"),
                "lease_ids": runtime.get("lease_ids", []),
                "replica_rank": replica_rank,
                "state": "LB_READY",
                "released": False,
                "runtime": runtime,
                "registration": registration,
                "bootstrap": bootstrap,
                "ready": ready,
                "error": None,
            }

    def _maybe_run_d4_runtime_smoke(self, config) -> None:
        """Run the opt-in post-training D4 command-chain smoke.

        The hook intentionally runs after native training has loaded stable
        weights.  It is a real main_ppo/Ray call chain, while the normal GS
        path uses ``execute_replica_operation`` directly during a live task.
        """
        config_get = getattr(config, "get", None)
        multitask_config = config_get("multitask", {}) if callable(config_get) else getattr(config, "multitask", {})
        test_config = multitask_config.get("d4_runtime_test", {}) if multitask_config is not None else {}
        if not bool(test_config.get("enabled", False)):
            return
        scenario = str(test_config.get("scenario", "split"))
        rollouter = self.components["rollouter"]
        trainer = self.components["trainer"]
        prepared = ray.get(rollouter.prepare_d4_runtime_smoke.remote(scenario))
        if prepared.get("expected_failure"):
            raise RuntimeError(f"D4 smoke requires a success scenario: {scenario}")
        donor_ranks = [int(rank) for rank in prepared.get("sleeping", {}).get("replica_ranks", [])]
        suspended = False
        replica_rank = None
        cleanup_done = False
        try:
            if donor_ranks:
                ray.get(trainer.suspend_donors_for_borrow.remote(donor_ranks))
                suspended = True
            result = self.execute_replica_operation("create", prepared["spec"])
            replica_rank = result.get("replica_rank")
            if replica_rank is not None:
                replica_rank = int(replica_rank)
            if result.get("state") != "LB_READY":
                raise RuntimeError(f"D4 create chain did not reach LB_READY: {result}")
            probe = ray.get(rollouter.probe_replica_ready.remote(replica_rank))
            print(f"D4_RUNTIME_RESULT {json.dumps(result | {'probe': probe}, sort_keys=True, default=str)}")
            cleanup = ray.get(rollouter.cleanup_d4_runtime.remote(replica_rank))
            cleanup_done = True
            print(f"D4_RUNTIME_CLEANUP {json.dumps(cleanup, sort_keys=True, default=str)}")
        finally:
            if replica_rank is not None and not cleanup_done:
                try:
                    cleanup = ray.get(rollouter.cleanup_d4_runtime.remote(replica_rank))
                    print(f"D4_RUNTIME_CLEANUP {json.dumps(cleanup, sort_keys=True, default=str)}")
                except Exception:
                    logger.warning("D4 smoke cleanup failed for replica %s", replica_rank, exc_info=True)
            if suspended:
                ray.get(trainer.resume_donors_after_borrow.remote(donor_ranks))

    def _create_rollouter(self, config) -> None:
        """Preserve native main.py:117-136; replace only the type and GS argument."""
        print("[ASYNC MAIN] Starting create rollouter...")
        rollouter = MultiTaskFullyAsyncRollouter.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            processor=self.components["processor"],
            device_name=config.trainer.device,
            group_scheduler=self.group_scheduler,
        )

        if "hybrid_worker_group" in self.components:
            ray.get(rollouter.set_hybrid_worker_group.remote(self.components["hybrid_worker_group"]))
            print("[ASYNC MAIN] Hybrid worker group injected into rollouter")

        ray.get(rollouter.init_workers.remote())
        ray.get(rollouter.set_max_required_samples.remote())

        self.components["rollouter"] = rollouter
        print("[ASYNC MAIN] Rollouter created and initialized successfully")

    def _create_trainer(self, config) -> None:
        """Preserve native main.py:138-157; replace only the Trainer ActorClass."""
        print("[ASYNC MAIN] Starting create trainer...")
        trainer_role_mapping = {
            role: worker_cls
            for role, worker_cls in self.components["role_worker_mapping"].items()
            if role != Role.Rollout
        }

        trainer = MultiTaskFullyAsyncTrainer.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping=trainer_role_mapping,
            resource_pool_manager=create_resource_pool_manager(config, roles=list(trainer_role_mapping.keys())),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            device_name=config.trainer.device,
        )

        ray.get(trainer.init_workers.remote())
        self.components["trainer"] = trainer
        print("[ASYNC MAIN] FullyAsyncTrainer created and initialized successfully")
