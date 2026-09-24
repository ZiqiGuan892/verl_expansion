"""Native receiver extension with an opt-in per-parameter audit manifest."""

import hashlib
import os

import torch

from verl.checkpoint_engine.base import CheckpointEngineWorker
from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.rollout.utils import ensure_async_iterator


class MultiTaskCheckpointEngineWorker(CheckpointEngineWorker):
    """Record receiver-side parameter fingerprints while reusing native transport."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parameter_validation_enabled = (
            os.environ.get("MULTITASK_PARAMETER_VALIDATION", "0") == "1"
            or os.environ.get("MULTITASK_SOURCE_VALIDATION", "0") == "1"
        )
        self._last_parameter_manifest = {
            "complete": False,
            "global_steps": None,
            "wire_format": None,
            "parameters": [],
            "parameter_count": 0,
            "total_numel": 0,
        }

    @staticmethod
    def _tensor_sha256(tensor: torch.Tensor) -> str:
        # Hash the exact contiguous byte representation without moving the
        # tensor payload back to the driver or retaining a second model copy.
        raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
        return hashlib.sha256(raw).hexdigest()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, global_steps: int = None):
        """Reuse native receive/load flow and retain one entry per parameter."""
        if not self.parameter_validation_enabled:
            return await super().update_weights(global_steps=global_steps)
        wire_format = getattr(self.checkpoint_engine, "wire_format", "named_tensors")
        manifest = {
            "complete": False,
            "global_steps": global_steps,
            "wire_format": wire_format,
            "parameters": [],
            "parameter_count": 0,
            "total_numel": 0,
        }
        if wire_format != "named_tensors":
            self._last_parameter_manifest = manifest
            raise NotImplementedError(
                f"per-parameter validation currently requires wire_format='named_tensors', got {wire_format!r}"
            )

        received = self.checkpoint_engine.receive_weights(global_steps=global_steps)

        async def audited_weights():
            async for name, tensor in ensure_async_iterator(received):
                manifest["parameters"].append(
                    {
                        "name": str(name),
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                        "numel": int(tensor.numel()),
                        "sha256": self._tensor_sha256(tensor),
                    }
                )
                manifest["parameter_count"] += 1
                manifest["total_numel"] += int(tensor.numel())
                yield name, tensor

        try:
            await self.server_adapter.update_weights(
                audited_weights(),
                global_steps=global_steps,
                wire_format=wire_format,
            )
            manifest["complete"] = True
            self._last_parameter_manifest = manifest
        except Exception:
            self._last_parameter_manifest = manifest
            raise

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_parameter_manifest(self) -> dict:
        """Return only the last receiver-side audit metadata."""
        return self._last_parameter_manifest
