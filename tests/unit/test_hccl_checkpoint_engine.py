"""HCCL finalize regression tests with explicit CPU-only dependency substitutes.

These exercise the actual plugin module but do not load Ray, torch_npu or CANN.
Real communicator destruction/recreation must be validated by D3_test.sh.
"""

import importlib.util
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


SOURCE = Path(__file__).resolve().parents[2] / "src/multi_task_scheduler/checkpoint/hccl_checkpoint_engine.py"


@pytest.fixture
def backend(monkeypatch):
    registry = {"nccl": object()}

    def register(name):
        def decorator(cls):
            registry[name] = cls
            return cls
        return decorator

    class NativeHCCL:
        def prepare(self):
            pass

        def init_process_group(self):
            pass

        async def send_weights(self, *args, **kwargs):
            pass

        def receive_weights(self):
            pass

    npu = SimpleNamespace(
        device=Mock(side_effect=lambda device: nullcontext()),
        synchronize=Mock(),
        empty_cache=Mock(),
    )
    module_attributes = {
        "torch": {"npu": npu, "Tensor": object, "uint8": object},
        "verl": {"__path__": []},
        "verl.checkpoint_engine": {"__path__": []},
        "verl.checkpoint_engine.base": {"CheckpointEngineRegistry": SimpleNamespace(register=register)},
        "verl.checkpoint_engine.hccl_checkpoint_engine": {"HCCLCheckpointEngine": NativeHCCL},
        "verl.workers": {"__path__": []},
        "verl.workers.rollout": {"__path__": []},
        "verl.workers.rollout.utils": {
            "ensure_async_iterator": None,
        },
    }
    async def ensure_async_iterator(iterable):
        if hasattr(iterable, "__aiter__"):
            async for item in iterable:
                yield item
        else:
            for item in iterable:
                yield item

    module_attributes["verl.workers.rollout.utils"]["ensure_async_iterator"] = ensure_async_iterator
    for name, attrs in module_attributes.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("test_hccl_plugin", SOURCE)
    plugin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin)
    engine = plugin.MultiTaskHCCLCheckpointEngine()
    engine.rebuild_group = True
    engine.rank = 1
    engine.world_size = 3
    engine.device = 2
    engine.send_buf, engine.recv_buf = object(), object()
    communicator = SimpleNamespace(comm=object(), hccl=SimpleNamespace(hcclCommDestroy=Mock()))
    engine.pyhccl = communicator
    return engine, communicator, npu, registry, NativeHCCL


def test_current_ascend_api_destroys_once_and_releases_buffers(backend):
    engine, communicator, npu, _, _ = backend
    events = Mock()
    events.attach_mock(npu.synchronize, "synchronize")
    events.attach_mock(communicator.hccl.hcclCommDestroy, "destroy")
    events.attach_mock(npu.empty_cache, "empty_cache")

    # Reproduces the server API: communicator has no destroyComm method.
    engine.finalize()
    engine.finalize()

    communicator.hccl.hcclCommDestroy.assert_called_once_with(communicator.comm)
    npu.device.assert_called_once_with(2)
    assert [call[0] for call in events.mock_calls] == ["synchronize", "destroy", "empty_cache", "empty_cache"]
    assert engine.pyhccl is None
    assert engine.rank is None and engine.world_size is None
    assert engine.send_buf is None and engine.recv_buf is None


def test_legacy_communicator_api_remains_supported(backend):
    engine, communicator, _, _, _ = backend
    communicator.destroyComm = Mock()
    engine.finalize()
    communicator.destroyComm.assert_called_once_with(communicator.comm)
    communicator.hccl.hcclCommDestroy.assert_not_called()


def test_nonparticipating_training_rank_needs_no_destroy(backend):
    engine, _, npu, _, _ = backend
    engine.rank = -1
    engine.pyhccl = None
    engine.finalize()
    npu.synchronize.assert_not_called()
    assert engine.rank is None and engine.world_size is None
    assert engine.send_buf is None and engine.recv_buf is None


def test_rebuild_disabled_preserves_communicator(backend):
    engine, communicator, npu, _, _ = backend
    engine.rebuild_group = False
    engine.finalize()
    assert engine.pyhccl is communicator
    assert engine.rank == 1 and engine.world_size == 3
    communicator.hccl.hcclCommDestroy.assert_not_called()
    npu.synchronize.assert_not_called()
    assert engine.send_buf is None and engine.recv_buf is None


def test_destroy_failure_is_propagated_without_losing_handle(backend):
    engine, communicator, npu, _, _ = backend
    communicator.hccl.hcclCommDestroy.side_effect = RuntimeError("HCCL destroy failed")
    buffers = engine.send_buf, engine.recv_buf
    with pytest.raises(RuntimeError, match="HCCL destroy failed"):
        engine.finalize()
    assert engine.pyhccl is communicator
    assert engine.rank == 1 and engine.world_size == 3
    assert (engine.send_buf, engine.recv_buf) == buffers
    npu.empty_cache.assert_not_called()


def test_plugin_registers_separate_backend_and_inherits_transfer(backend):
    engine, _, _, registry, native = backend
    assert registry["multitask_hccl"] is type(engine)
    assert registry["nccl"] is not type(engine)
    for method in ("prepare", "init_process_group", "receive_weights"):
        assert getattr(type(engine), method) is getattr(native, method)
    assert getattr(type(engine), "send_weights") is not getattr(native, "send_weights")
