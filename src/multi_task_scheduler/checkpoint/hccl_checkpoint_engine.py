"""Opt-in HCCL backend for vLLM-Ascend communicator teardown compatibility.

Load through verl's ``checkpoint_engine.custom_backend_module`` on both the
training and receiving workers.  The native ``nccl`` registry entry is unchanged.
"""

import torch

from verl.checkpoint_engine.base import CheckpointEngineRegistry
from verl.checkpoint_engine.hccl_checkpoint_engine import HCCLCheckpointEngine


@CheckpointEngineRegistry.register("multitask_hccl")
class MultiTaskHCCLCheckpointEngine(HCCLCheckpointEngine):
    """Inherit native HCCL transfer; adapt only communication-group teardown."""

    def finalize(self) -> None:
        """Destroy the owned communicator once, then release transfer buffers.

        vLLM-Ascend exposes destruction on the HCCL library wrapper; some older
        integrations expose ``destroyComm`` on the communicator itself.  A
        failed destroy must propagate and retain the handle for diagnosis/retry.
        """
        if self.rebuild_group:
            communicator = self.pyhccl
            if communicator is not None:
                # HCCL operations are asynchronous.  Wait on the owning device
                # before destroying its communicator or freeing weight buckets.
                with torch.npu.device(self.device):
                    torch.npu.synchronize()
                    destroy = getattr(communicator, "destroyComm", None)
                    if callable(destroy):
                        destroy(communicator.comm)
                    else:
                        communicator.hccl.hcclCommDestroy(communicator.comm)
                self.pyhccl = None
            # Non-sending training ranks have no communicator.  Checking the
            # handle also makes repeated finalize safe after rank becomes None.
            self.rank = None
            self.world_size = None

        self.send_buf = None
        self.recv_buf = None
        torch.npu.empty_cache()
