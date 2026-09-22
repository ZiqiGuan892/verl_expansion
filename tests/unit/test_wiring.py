"""Scoped AST execution with mocked parent/runtime boundaries.

These tests exercise real override bodies and argument forwarding. They do not
import verl, instantiate its real parents, or claim Ray/GPU runtime success.
The separate native_unit layer covers imports and actual parent relationships.
"""

import ast
import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

import pytest


SOURCE = Path(__file__).resolve().parents[2] / "src/multi_task_scheduler"
INTEGRATION = "integration/verl/experimental_fully_async"


def _isolated_class(relative, name, parent, **globals_for_test):
    """Execute a class body with a clearly substituted parent, without import patches."""
    path = SOURCE / relative
    parsed = ast.parse(path.read_text())
    node = next(item for item in parsed.body if isinstance(item, ast.ClassDef) and item.name == name)
    node.bases = [ast.Name(id="TestParent", ctx=ast.Load())]
    node.decorator_list = []
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node
    ], type_ignores=[])
    scope = {"TestParent": parent, **globals_for_test}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return scope[name]


def test_task_runner_creation_preserves_roles_arguments_and_initialization_order():
    events = []

    def rpc(name):
        return SimpleNamespace(remote=Mock(side_effect=lambda: events.append(name)))

    trainer = SimpleNamespace(init_workers=rpc("trainer.init_workers"))
    rollouter = SimpleNamespace(init_workers=rpc("rollouter.init_workers"),
                               set_max_required_samples=rpc("rollouter.set_max_required_samples"))
    trainer_class = SimpleNamespace(remote=Mock(return_value=trainer))
    rollouter_class = SimpleNamespace(remote=Mock(return_value=rollouter))
    role = SimpleNamespace(Rollout=object())
    resource_pool = object()
    pool_factory = Mock(return_value=resource_pool)
    test_class = _isolated_class(
        f"{INTEGRATION}/task_runner.py", "MultiTaskFullyAsyncTaskRunner", object,
        ray=SimpleNamespace(get=lambda result: result), Role=role,
        MultiTaskFullyAsyncTrainer=trainer_class, MultiTaskFullyAsyncRollouter=rollouter_class,
        create_resource_pool_manager=pool_factory,
    )
    runner = test_class()
    runner.group_scheduler = object()
    runner.components = {"tokenizer": object(), "processor": object(), "ray_worker_group_cls": object(),
                         "role_worker_mapping": {"actor": object(), role.Rollout: object()}}
    config = SimpleNamespace(trainer=SimpleNamespace(device="cuda"))
    runner._create_trainer(config)
    runner._create_rollouter(config)
    trainer_class.remote.assert_called_once_with(
        config=config, tokenizer=runner.components["tokenizer"],
        role_worker_mapping={"actor": runner.components["role_worker_mapping"]["actor"]},
        resource_pool_manager=resource_pool, ray_worker_group_cls=runner.components["ray_worker_group_cls"],
        device_name="cuda",
    )
    pool_factory.assert_called_once_with(config, roles=["actor"])
    rollouter_class.remote.assert_called_once_with(
        config=config, tokenizer=runner.components["tokenizer"], processor=runner.components["processor"],
        device_name="cuda", group_scheduler=runner.group_scheduler,
    )
    assert events == ["trainer.init_workers", "rollouter.init_workers", "rollouter.set_max_required_samples"]
    assert runner.components["trainer"] is trainer
    assert runner.components["rollouter"] is rollouter


@pytest.mark.parametrize("native_failure", [False, True])
def test_runner_attaches_real_chain_reference_before_native_run_and_detaches_after(native_failure):
    events = []
    task_handle = object()  # Explicit runtime substitute, not a Ray ActorHandle.
    config = object()

    class Parent:
        def __init__(self):
            self.components = {}

        def run(self, received):
            assert received is config
            events.append("native_run")
            if native_failure:
                raise RuntimeError("native failure")
            return "native result"

    scheduler = SimpleNamespace(
        attach_task=SimpleNamespace(remote=Mock(side_effect=lambda *args: events.append("attach"))),
        detach_task=SimpleNamespace(remote=Mock(side_effect=lambda *args: events.append("detach"))),
    )
    runner_class = _isolated_class(
        f"{INTEGRATION}/task_runner.py", "MultiTaskFullyAsyncTaskRunner", Parent,
        ray=SimpleNamespace(get=lambda result, **kwargs: result,
                            get_runtime_context=lambda: SimpleNamespace(get_actor_id=lambda: "task-a",
                                                                         current_actor=task_handle)),
        get_or_create_group_scheduler=lambda: scheduler, logger=logging.getLogger(__name__),
    )
    runner = runner_class()
    if native_failure:
        with pytest.raises(RuntimeError, match="native failure"):
            runner.run(config)
    else:
        assert runner.run(config) == "native result"
    assert events == ["attach", "native_run", "detach"]
    scheduler.attach_task.remote.assert_called_once_with("task-a", task_handle)
    scheduler.detach_task.remote.assert_called_once_with("task-a")
    assert runner.group_scheduler is scheduler


def test_manager_preselects_replica_and_forwards_native_load_balancer_flags():
    replica_class = object()
    load_balancer_class = object()
    forwarded = []

    class Parent:
        def __init__(self, *args):
            assert self.rollout_replica_class is replica_class
            forwarded.append(args)

    actor_class = SimpleNamespace(remote=Mock(return_value=object()))
    ray_substitute = SimpleNamespace(remote=Mock(return_value=actor_class))
    manager_class = _isolated_class(
        f"{INTEGRATION}/llm_server_manager.py", "MultiTaskLLMServerManager", Parent,
        MultiTaskvLLMReplica=replica_class, MultiTaskGlobalRequestLoadBalancer=load_balancer_class,
        DEFAULT_ROUTING_CACHE_SIZE=123, ray=ray_substitute, asyncio=asyncio,
    )
    config, pool, scheduler = object(), object(), object()
    manager = manager_class(config, rollout_resource_pool=pool, group_scheduler=scheduler)
    assert forwarded == [(config, None, pool)]
    manager.server_addresses = ["server-a", "server-b"]
    manager.server_handles = [object(), object()]
    manager.rollout_config = SimpleNamespace(full_determinism=True)
    asyncio.run(manager._init_global_load_balancer())
    ray_substitute.remote.assert_called_once_with(load_balancer_class)
    actor_class.remote.assert_called_once_with(
        servers=dict(zip(manager.server_addresses, manager.server_handles)), max_cache_size=123,
        full_determinism=True, group_scheduler=scheduler,
    )
    assert manager.global_load_balancer is actor_class.remote.return_value


def test_load_balancer_constructor_preserves_native_arguments_and_gs_handle():
    constructor = Mock()

    class Parent:
        def __init__(self, *args, **kwargs):
            constructor(*args, **kwargs)

    balancer_class = _isolated_class(
        "rollout/load_balancer.py", "MultiTaskGlobalRequestLoadBalancer", Parent, DEFAULT_ROUTING_CACHE_SIZE=123,
    )
    servers, scheduler = {"server-a": object()}, object()
    balancer = balancer_class(servers, max_cache_size=5, full_determinism=True, group_scheduler=scheduler)
    constructor.assert_called_once_with(servers, max_cache_size=5, full_determinism=True)
    assert balancer.group_scheduler is scheduler


def test_replica_selects_http_server_and_checkpoint_worker_without_creating_extra_runtime():
    constructor = Mock()

    class Parent:
        def __init__(self, *args, **kwargs):
            constructor(*args, **kwargs)
            self.config, self.model_config, self.replica_rank = object(), object(), 3

    server_class, worker_class = object(), object()
    remote_descriptors = [object(), object()]
    ray_substitute = SimpleNamespace(remote=Mock(side_effect=remote_descriptors))
    wrapper = Mock(return_value=object())
    replica_class = _isolated_class(
        "rollout/replica.py", "MultiTaskvLLMReplica", Parent, ray=ray_substitute,
        MultiTaskvLLMHttpServer=server_class, MultiTaskCheckpointEngineWorker=worker_class,
        RayClassWithInitArgs=wrapper,
    )
    replica = replica_class("model", replica_rank=3)
    constructor.assert_called_once_with("model", replica_rank=3)
    assert replica.server_class is remote_descriptors[0]
    assert replica.get_ray_class_with_init_args() is wrapper.return_value
    assert ray_substitute.remote.call_args_list == [call(server_class), call(worker_class)]
    wrapper.assert_called_once_with(
        cls=remote_descriptors[1], rollout_config=replica.config,
        model_config=replica.model_config, replica_rank=3,
    )


def test_rollouter_passes_gs_and_preserves_native_agent_loop_arguments():
    constructor = Mock()

    class Parent:
        def __init__(self, *args, **kwargs):
            constructor(*args, **kwargs)

    manager = SimpleNamespace(get_client=Mock(return_value=object()))
    manager_factory = SimpleNamespace(create=AsyncMock(return_value=manager))
    agent_factory = SimpleNamespace(create=AsyncMock(return_value=object()))
    client_class = object()
    rollouter_class = _isolated_class(
        f"{INTEGRATION}/rollouter.py", "MultiTaskFullyAsyncRollouter", Parent,
        MultiTaskLLMServerManager=manager_factory, FullyAsyncAgentLoopManager=agent_factory,
        FullyAsyncLLMServerClient=client_class,
    )
    config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(mode="async")))
    scheduler, tokenizer, processor = object(), object(), object()
    rollouter = rollouter_class(config, tokenizer, processor, "cuda", group_scheduler=scheduler)
    constructor.assert_called_once_with(config, tokenizer, processor=processor, device_name="cuda")
    rollouter.config, rollouter.use_rm = config, False
    rollouter.reward_loop_manager = SimpleNamespace(reward_loop_workers=[object()])
    rollouter.teacher_model_manager = SimpleNamespace(get_client=Mock(return_value=object()))
    rollouter.get_hybrid_worker_group = Mock(return_value=None)
    asyncio.run(rollouter._init_async_rollout_manager())
    manager_factory.create.assert_awaited_once_with(config=config, worker_group=None, group_scheduler=scheduler)
    manager.get_client.assert_called_once_with(client_cls=client_class)
    agent_factory.create.assert_awaited_once_with(
        config=config, llm_client=manager.get_client.return_value,
        reward_loop_worker_handles=rollouter.reward_loop_manager.reward_loop_workers,
        teacher_client=rollouter.teacher_model_manager.get_client.return_value,
    )
    assert rollouter.llm_server_manager is manager
    assert rollouter.async_rollout_manager is agent_factory.create.return_value
    assert rollouter.async_rollout_mode is True


def test_trainer_uses_rollouter_replica_projection_for_native_checkpoint_manager():
    checkpoint_config = SimpleNamespace(backend="nccl")
    converter = Mock(return_value=checkpoint_config)
    factory = Mock(return_value=object())
    trainer_class = _isolated_class(
        f"{INTEGRATION}/trainer.py", "MultiTaskFullyAsyncTrainer", object,
        omega_conf_to_dataclass=converter, MultiTaskCheckpointEngineManager=factory,
    )
    trainer = trainer_class()
    trainer.config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(checkpoint_engine=object())))
    trainer.actor_wg = object()
    replicas = [object(), object()]
    trainer.rollouter = SimpleNamespace(get_replicas=SimpleNamespace(remote=AsyncMock(return_value=replicas)))
    asyncio.run(trainer._setup_checkpoint_manager())
    converter.assert_called_once_with(trainer.config.actor_rollout_ref.rollout.checkpoint_engine)
    factory.assert_called_once_with(config=checkpoint_config, actor_wg=trainer.actor_wg, replicas=replicas)
    assert trainer.checkpoint_manager is factory.return_value
    assert not hasattr(trainer, "group_scheduler")
