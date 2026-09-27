"""One explicit runtime selection; disabled imports never load Ray or verl."""

from collections.abc import Mapping

PROFILE_ID = "experimental_fully_async_standalone"
_MISSING = object()


class ProfileConfigurationError(ValueError):
    """The requested profile cannot use this pure-STANDALONE adapter family."""


def _select(config, path, default=_MISSING):
    value = config
    for key in path.split("."):
        if value is None and default is not _MISSING:
            return default
        if not isinstance(value, Mapping):
            raise ProfileConfigurationError(f"{path}: parent of {key} must be a mapping")
        if key not in value:
            if default is _MISSING:
                raise ProfileConfigurationError(f"Missing configuration: {path}")
            return default
        value = value[key]
    return value


def validate_runtime_profile(config) -> bool:
    """Check selection and minimal native preconditions without changing config.

    Native main has already mapped rollout node/GPU counts. Backend placement and
    training algorithm checks remain native; this is not a full topology validator.
    """
    if not isinstance(config, Mapping):
        raise ProfileConfigurationError("Configuration must be a mapping")
    profile = _select(config, "multitask.runtime.profile", None)
    if profile is None:
        return False
    if not isinstance(profile, str) or profile != PROFILE_ID:
        raise ProfileConfigurationError(f"Unknown multitask.runtime.profile: {profile!r}")
    if set(_select(config, "multitask.runtime")) != {"profile"}:
        raise ProfileConfigurationError("multitask.runtime accepts only profile, not per-class selectors")
    if _select(config, "multitask.rollout_deployment", None) is not None:
        raise ProfileConfigurationError("Use only multitask.runtime.profile to select the deployment")
    for path, expected in (
        ("actor_rollout_ref.hybrid_engine", False),
        ("async_training.use_trainer_do_validate", False),
        ("async_training.use_dynamic_resource_scheduling", False),
        ("actor_rollout_ref.rollout.mode", "async"),
        ("actor_rollout_ref.rollout.name", "vllm"),
        ("actor_rollout_ref.rollout.calculate_log_probs", True),
        ("data.train_batch_size", 0),
        ("data.gen_batch_size", 1),
    ):
        value = _select(config, path)
        if type(value) is not type(expected) or value != expected:
            raise ProfileConfigurationError(f"{path} must be {expected!r}, got {value!r}")
    prefix = "actor_rollout_ref.rollout"
    disaggregation = _select(config, f"{prefix}.disaggregation", None)
    if disaggregation is not None:
        if not isinstance(disaggregation, Mapping) or disaggregation.get("enabled", False) is not False:
            raise ProfileConfigurationError("This vLLM adapter family does not implement PD replicas")
    backend = _select(config, f"{prefix}.checkpoint_engine.backend")
    if not isinstance(backend, str) or not backend.strip() or backend == "naive":
        raise ProfileConfigurationError("Pure STANDALONE requires a non-naive checkpoint engine")
    sizes = {}
    for field in (
        "nnodes", "n_gpus_per_node", "tensor_model_parallel_size",
        "data_parallel_size", "pipeline_model_parallel_size",
    ):
        value = _select(config, f"{prefix}.{field}")
        if type(value) is not int or value <= 0:
            raise ProfileConfigurationError(f"{prefix}.{field} must be a positive integer")
        sizes[field] = value
    for field in ("nnodes", "n_gpus_per_node"):
        value = _select(config, f"rollout.{field}")
        if type(value) is not int or value != sizes[field]:
            raise ProfileConfigurationError(f"Native main must map rollout.{field} before selecting MultiTask")
    replica_gpus = (
        sizes["tensor_model_parallel_size"] * sizes["data_parallel_size"] * sizes["pipeline_model_parallel_size"]
    )
    if sizes["nnodes"] * sizes["n_gpus_per_node"] < replica_gpus:
        raise ProfileConfigurationError("Rollout resources cannot fit one replica")
    return True


def resolve_runtime_profile(config):
    """Return the real root ActorClass, or None for a disabled profile.

    Import failures propagate: an enabled MultiTask run must not silently fall
    back to native classes. This function never starts Ray or discovers GS.
    """
    from multi_task_scheduler.testing.launch_check import verify_config

    verify_config(config)
    if not validate_runtime_profile(config):
        return None
    from .experimental_fully_async.task_runner import MultiTaskFullyAsyncTaskRunner

    return MultiTaskFullyAsyncTaskRunner
