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

"""Keep native training while exposing CE membership/bootstrap boundaries."""

import asyncio
import json

import ray

from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
from verl.utils.config import omega_conf_to_dataclass

from multi_task_scheduler.checkpoint.checkpoint_engine_manager import MultiTaskCheckpointEngineManager
from multi_task_scheduler.integration.verl.ray_actor import unwrap_native_actor_class


@ray.remote(num_cpus=10)
class MultiTaskFullyAsyncTrainer(unwrap_native_actor_class(FullyAsyncTrainer)):
    """Own the CE Manager in this Actor; inherit training and weight synchronization."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parameter_snapshot_gate = asyncio.Lock()
        self.parameter_validation_enabled = self._read_parameter_validation_enabled(self.config)
        self._training_completion = None
        self._d3_bootstrap_rank = None
        self._d3_cleanup_after_test = True
        self._d3_bootstrap_result = None
        self._d3_suspended_donor_ranks = []

    @staticmethod
    def _read_parameter_validation_enabled(config) -> bool:
        """Read the opt-in, test-only CE per-parameter verification switch."""
        config_get = getattr(config, "get", None)
        multitask = config_get("multitask", {}) if callable(config_get) else getattr(config, "multitask", {})
        validation = multitask.get("parameter_validation", {}) if multitask is not None else {}
        enabled = validation.get("enabled", False) if hasattr(validation, "get") else False
        return bool(enabled)

    def get_training_completion(self) -> dict:
        """Return the only authoritative training-completion evidence.

        ``FullyAsyncTaskRunner`` waits for both rollouter and trainer actors,
        but its generic component messages do not identify whether all planned
        training steps were processed.  The progress bar is advanced by the
        native trainer exactly once per completed training step, so this
        projection compares that count with the rollouter-derived target.
        """
        target_steps = self.total_train_steps
        progress = getattr(self, "progress_bar", None)
        completed_steps = int(getattr(progress, "n", 0)) if progress is not None else 0
        target = int(target_steps) if target_steps is not None else None
        completed = target is not None and completed_steps == target
        return {
            "state": "COMPLETED" if completed else "INCOMPLETE",
            "completed": bool(completed),
            "completed_steps": completed_steps,
            "target_steps": target,
            "global_steps": int(getattr(self, "global_steps", 0)),
            "current_param_version": int(getattr(self, "current_param_version", 0)),
        }

    async def fit(self):
        """Run native training and emit a strict completion receipt."""
        try:
            result = await super().fit()
        except Exception as exc:
            self._training_completion = self.get_training_completion()
            self._training_completion.update({"state": "FAILED", "error": str(exc)})
            raise
        self._training_completion = self.get_training_completion()
        print(
            "MULTITASK_TRAINING_COMPLETE "
            f"{json.dumps(self._training_completion, sort_keys=True, default=str)}",
            flush=True,
        )
        if not self._training_completion["completed"]:
            raise RuntimeError(f"training returned before all planned steps completed: {self._training_completion}")
        return result

    async def _setup_checkpoint_manager(self):
        """Preserve native trainer.py:217-224; replace only the Manager class."""
        replicas = await self.rollouter.get_replicas.remote()
        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        self.checkpoint_manager = MultiTaskCheckpointEngineManager(
            config=checkpoint_engine_config, actor_wg=self.actor_wg, replicas=replicas
        )
        self.checkpoint_manager.parameter_validation_enabled = self.parameter_validation_enabled
        print(f"[FullyAsyncTrainer] Checkpoint manager initialized (backend={checkpoint_engine_config.backend})")

    async def register_replica(self, replica_rank: int) -> dict:
        """Register a manager-owned replica as pending CE membership."""
        replica = await self.rollouter.get_borrowed_replica_for_ce.remote(replica_rank)
        result = await self.checkpoint_manager.register_replica(replica)
        return result

    async def bootstrap_replica(self, replica_rank: int) -> dict:
        """Bootstrap one replica from one stable snapshot of current parameters."""
        replica = await self.rollouter.get_borrowed_replica_for_ce.remote(replica_rank)
        async with self.parameter_snapshot_gate:
            snapshot_version = int(self.current_param_version)
            result = await self.checkpoint_manager.bootstrap_replica(replica, snapshot_version)
            await self.rollouter.mark_replica_serving_version.remote(replica_rank, snapshot_version)
            return result

    async def unregister_replica(self, replica_rank: int) -> dict:
        """Remove one replica from future CE snapshots without destroying actors."""
        replica = await self.rollouter.get_borrowed_replica_for_ce.remote(replica_rank)
        return await self.checkpoint_manager.unregister_replica(replica)

    async def suspend_donors_for_borrow(self, replica_ranks) -> dict:
        """Remove donor ranks from CE effective membership for a borrow window.

        This is the CE-side hook for an externally-owned donor sleep
        transaction. It does not drain requests, change LB routing, sleep a
        vLLM server, or finalize an existing communication domain.
        """
        # TODO(lifecycle): call this before donor sleep and require the old CE
        # domain to be finalized before borrower creation.
        return await self.checkpoint_manager.suspend_replicas_for_sync(replica_ranks)

    async def resume_donors_after_borrow(self, replica_ranks) -> dict:
        """Re-enable donor CE membership after an externally-owned wake.

        The caller must already have completed server wake, target-only
        synchronization to the current actor version, and communication-domain
        finalization. No server or parameter operation is performed here.
        """
        # TODO(lifecycle): require and validate the wake/target-sync receipt
        # before making the donor effective again.
        return await self.checkpoint_manager.resume_replicas_for_sync(replica_ranks)

    async def load_checkpoint(self):
        """Load native checkpoints, then optionally run the D3 bootstrap hook."""
        result = await super().load_checkpoint()
        await self._maybe_run_d3_bootstrap_smoke()
        return result

    async def _maybe_run_d3_bootstrap_smoke(self) -> None:
        config_get = getattr(self.config, "get", None)
        multitask_config = config_get("multitask", {}) if callable(config_get) else getattr(self.config, "multitask", {})
        test_config = multitask_config.get("d3_bootstrap_test", {}) if multitask_config is not None else {}
        if not bool(test_config.get("enabled", False)) or self._d3_bootstrap_rank is not None:
            return
        scenario = str(test_config.get("scenario", "split"))
        self._d3_cleanup_after_test = bool(test_config.get("cleanup_after_test", True))
        prepared = None
        replica_rank = None
        registered = False
        try:
            prepared = await self.rollouter.run_d3_runtime_smoke.remote(scenario)
            replica_rank = int(prepared["replica_rank"])
            registration = await self.register_replica(replica_rank)
            registered = registration.get("state") in {"REGISTERED", "ALREADY_REGISTERED"}
            bootstrap = await self.bootstrap_replica(replica_rank)
            donor_ranks = [int(rank) for rank in prepared.get("sleeping_donor_ranks", [])]
            # The smoke fixture already slept donors to make device memory
            # available.  The only lifecycle step implemented here is the CE
            # projection change that prevents duplicate HCCL devices.
            # TODO(lifecycle): production TaskRunner ordering must be
            # lifecycle gate -> CE suspend/finalize -> drain/LB removal ->
            # server sleep -> borrowed creation. Do not copy this test order.
            memory = await self.suspend_donors_for_borrow(donor_ranks)
            memory.update(
                {
                    "replica_rank": replica_rank,
                    "state": "DONORS_SLEEPING_BORROWER_ONLY_EFFECTIVE",
                }
            )
            self._d3_suspended_donor_ranks = donor_ranks
            print(f"D3_MEMORY_RESULT {json.dumps(memory, sort_keys=True)}")
        except Exception:
            # The hook owns this test runtime.  Do best-effort local cleanup so
            # a failed bootstrap cannot leave vLLM/CE actors consuming the
            # next D3 attempt.  Production lifecycle recovery is still outside
            # D3 and is not invoked by ordinary training.
            if registered and replica_rank is not None:
                try:
                    await self.unregister_replica(replica_rank)
                except Exception:
                    pass
            if prepared is not None and replica_rank is not None:
                try:
                    await self.rollouter.cleanup_d3_runtime.remote(replica_rank, self.current_param_version)
                except Exception:
                    pass
            if self._d3_suspended_donor_ranks:
                try:
                    await self.resume_donors_after_borrow(self._d3_suspended_donor_ranks)
                except Exception:
                    pass
                self._d3_suspended_donor_ranks = []
            raise
        self._d3_bootstrap_rank = replica_rank
        self._d3_bootstrap_result = {
            "scenario": scenario,
            "replica_rank": replica_rank,
            "registration": registration,
            "bootstrap": bootstrap,
            "memory": memory,
            "bootstrap_version": bootstrap["version"],
            "state": "WEIGHTS_READY",
        }
        print(
            "D3_BOOTSTRAP_RESULT "
            f"{json.dumps(self._d3_bootstrap_result, sort_keys=True, default=str)}"
        )

    async def _fit_update_weights(self) -> dict | None:
        """Serialize normal sync with target bootstrap and verify one full sync."""
        async with self.parameter_snapshot_gate:
            result = await super()._fit_update_weights()
        if result is not None and self._d3_bootstrap_rank is not None:
            rank = self._d3_bootstrap_rank
            version = self.checkpoint_manager.last_synced_versions.get(rank)
            full_sync = {
                "replica_rank": rank,
                "state": "FULL_SYNC_READY" if version == self.current_param_version else "FAILED",
                "version": version,
                "expected_version": self.current_param_version,
            }
            print(f"D3_NORMAL_SYNC_RESULT {json.dumps(full_sync, sort_keys=True)}")
            if full_sync["state"] != "FULL_SYNC_READY":
                raise RuntimeError(f"D3 full sync did not update borrowed replica: {full_sync}")
            if self._d3_cleanup_after_test:
                await self.unregister_replica(rank)
                # cleanup_d3_runtime is a test fixture: it wakes the donor and
                # restores only version metadata. Production wake must first
                # run target-only CE synchronization/finalize, then resume
                # the donor in the effective set and finally publish LB READY.
                cleanup = await self.rollouter.cleanup_d3_runtime.remote(rank, self.current_param_version)
                await self.resume_donors_after_borrow(self._d3_suspended_donor_ranks)
                self._d3_bootstrap_result["cleanup"] = cleanup
                self._d3_suspended_donor_ranks = []
            self._d3_bootstrap_rank = None
        return result
