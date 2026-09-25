"""Opt-in acceptance teardown; this is not a lease/reclaim lifecycle API.

Call capture_runtime after engine startup, with generation/creation quiescent.
The owner must remove CE/LB membership before cleanup and restore donors only
after every borrower has a successful receipt. Nothing here changes CE, LB,
donor memory, placement groups, or GS claims. Ray and psutil are imported only
inside invoked helpers, so importing this module creates no distributed work.
"""

import asyncio
import copy
import json
import math
import time


class RuntimeCleanupError(RuntimeError):
    """A failed test cleanup with serializable, independently observed evidence."""

    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        super().__init__("test runtime cleanup unconfirmed: " + json.dumps(diagnostics, sort_keys=True, default=str))


def _id_text(value):
    return value.hex() if callable(getattr(value, "hex", None)) else str(value)


def _owned_handles(replica):
    if getattr(replica, "allocation_kind", None) != "borrowed" or getattr(replica, "owns_resource_pool", True):
        raise ValueError("test cleanup requires a borrowed replica that does not own the donor resource pool")
    pairs = [(handle, "server") for handle in replica.servers] + [(handle, "worker") for handle in replica.workers]
    if not replica.servers or not replica.workers:
        raise ValueError("capture requires a fully created borrowed runtime with HTTP and CE actors")
    handles = {_id_text(handle._actor_id): (handle, role) for handle, role in pairs}
    if len(handles) != len(pairs):
        raise ValueError("runtime contains duplicate owned actor handles")
    return handles


def _process_identity(process):
    return {"pid": process.pid, "create_time": process.create_time()}


def _capture_actor(actor, role):
    import os

    import psutil
    import ray

    context = ray.get_runtime_context()
    process = psutil.Process(os.getpid())
    identity = _process_identity(process)
    children = []
    endpoint = None
    if role == "server":
        # A stable pair of complete recursive enumerations is required. If a
        # child exits during enumeration, fail rather than silently dropping it.
        children = [_process_identity(child) for child in process.children(recursive=True)]
        repeated = [_process_identity(child) for child in process.children(recursive=True)]
        key = lambda item: (item["pid"], item["create_time"])
        if sorted(children, key=key) != sorted(repeated, key=key):
            raise RuntimeError("HTTP engine process tree changed during capture; runtime must be quiescent")
        if not children:
            raise RuntimeError("no HTTP engine child process was captured; mp-engine release cannot be verified")
        if actor.node_rank == 0:
            host, port = actor.get_server_address()
            endpoint = {"host": str(host).strip("[]"), "port": int(port)}
    return {
        "actor_id": _id_text(context.get_actor_id()),
        "node_id": _id_text(context.get_node_id()),
        "role": role,
        **identity,
        "children": children,
        "child_tree_complete": role == "server",
        "endpoint": endpoint,
    }


async def capture_runtime(replica) -> dict:
    """Snapshot actual actor identities and engine descendants before killing.

    Only a successful snapshot can authorize cleanup_runtime. A failed capture
    is not evidence of release, including when an actor is already unreachable.
    """
    handles = _owned_handles(replica)
    actors = await asyncio.wait_for(
        asyncio.gather(*(handle.__ray_call__.remote(_capture_actor, role) for handle, role in handles.values())),
        timeout=30.0,
    )
    snapshot = {"replica_rank": replica.replica_rank, "actors": actors, "captured_at": time.time()}
    _validate_snapshot(replica, snapshot)
    return snapshot


def _validate_snapshot(replica, snapshot):
    handles = _owned_handles(replica)
    if snapshot.get("replica_rank") != replica.replica_rank:
        raise ValueError("cleanup snapshot belongs to a different replica")
    actors = snapshot.get("actors", [])
    if len(actors) != len(handles) or {actor["actor_id"] for actor in actors} != set(handles):
        raise ValueError("cleanup snapshot must contain exactly this replica's owned actor IDs")
    roots = {(actor["node_id"], actor["pid"]) for actor in actors}
    endpoints = 0
    for actor in actors:
        if actor["role"] != handles[actor["actor_id"]][1] or not actor["node_id"]:
            raise ValueError("snapshot actor role/node does not match its ownership")
        if actor["role"] == "server" and (not actor.get("child_tree_complete") or not actor.get("children")):
            raise ValueError("HTTP engine process-tree evidence is missing")
        if actor["role"] != "server" and actor.get("children"):
            raise ValueError("only captured HTTP engine children may be terminated")
        for identity in [actor] + actor.get("children", []):
            pid, started = identity["pid"], identity["create_time"]
            if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
                raise ValueError("invalid captured process PID")
            if (
                isinstance(started, bool) or not isinstance(started, (int, float))
                or not math.isfinite(started) or started <= 0
            ):
                raise ValueError("invalid captured process start time")
        for child in actor.get("children", []):
            if (actor["node_id"], child["pid"]) in roots:
                raise ValueError("an owned actor cannot also be an engine-child kill target")
        endpoint = actor.get("endpoint")
        if endpoint is not None:
            if actor["role"] != "server" or not isinstance(endpoint.get("host"), str) or not endpoint["host"]:
                raise ValueError("invalid captured HTTP endpoint host")
            port = endpoint.get("port")
            if isinstance(port, bool) or not isinstance(port, int) or not 0 < port < 65536:
                raise ValueError("invalid captured HTTP endpoint port")
            endpoints += 1
    if endpoints != 1:
        raise ValueError("expected exactly one primary HTTP endpoint in the runtime snapshot")
    return handles


def _merge_capture(snapshot, refreshed):
    """Keep descendants captured earlier even if they have since been orphaned."""
    latest = {actor["actor_id"]: actor for actor in refreshed["actors"]}
    merged = copy.deepcopy(snapshot)
    for actor in merged["actors"]:
        current = latest[actor["actor_id"]]
        for key in ("role", "node_id", "pid", "create_time", "endpoint"):
            if actor[key] != current[key]:
                raise RuntimeError(f"owned actor {actor['actor_id']} changed {key} since initial capture")
        children = {
            (child["pid"], child["create_time"]): child
            for child in actor["children"] + current["children"]
        }
        actor["children"] = [children[key] for key in sorted(children)]
    merged["refreshed_at"] = refreshed["captured_at"]
    return merged


def _inspect_process(identity, terminate=None):
    import os
    import sys

    import psutil

    try:
        process = psutil.Process(identity["pid"])
        if process.create_time() != identity["create_time"]:
            return {**identity, "gone": True, "exited": True, "reaped": True, "state": "PID_REUSED"}
        status = process.status()
        terminal = {getattr(psutil, "STATUS_ZOMBIE", "zombie"), getattr(psutil, "STATUS_DEAD", "dead")}
        thread_states = []
        if status in terminal:
            if not sys.platform.startswith("linux"):
                raise RuntimeError(f"terminal process-state verification is only implemented for Linux: {status}")
            # A Linux thread-group leader can be a zombie while other threads
            # still execute. Do not infer whole-process exit from its state
            # alone; every remaining task must also be terminal.
            for thread in process.threads():
                try:
                    thread_states.append({"tid": thread.id, "state": psutil.Process(thread.id).status()})
                except getattr(psutil, "ZombieProcess", ()):
                    raise RuntimeError(f"could not inspect captured process thread {thread.id}") from None
                except psutil.NoSuchProcess:
                    continue
            if all(thread["state"] in terminal for thread in thread_states):
                # Exit releases execution/address-space/file resources. An
                # unreaped zombie retains its process-table entry until its
                # parent/PID 1 waits; signaling it cannot make it more exited.
                return {
                    **identity, "gone": False, "exited": True, "reaped": False,
                    "state": status, "thread_states": thread_states, "signal": None,
                }
        if terminate:
            if process.pid == os.getpid():
                raise RuntimeError("observer may not terminate itself")
            # psutil's signaling methods also check process identity, protecting
            # against PID reuse between the comparison and the signal.
            if terminate == "kill":
                process.kill()
            else:
                process.terminate()
        return {
            **identity, "gone": False, "exited": False, "reaped": False,
            "state": status, "thread_states": thread_states, "signal": terminate,
        }
    except getattr(psutil, "ZombieProcess", ()) as exc:
        # ZombieProcess subclasses NoSuchProcess in psutil. It does not prove
        # disappearance, so never let the generic handler label it reaped.
        raise RuntimeError(f"could not verify terminal process identity {identity}: {exc}") from exc
    except psutil.NoSuchProcess:
        return {**identity, "gone": True, "exited": True, "reaped": True, "state": "NOT_FOUND"}


def _inspect_node(actors, terminate=None):
    import errno
    import socket

    import ray

    node_id = _id_text(ray.get_runtime_context().get_node_id())
    if any(actor["node_id"] != node_id for actor in actors):
        raise RuntimeError("cleanup observer ran on a different node from the captured processes")
    processes, children, endpoints = [], [], []
    for actor in actors:
        processes.append(_inspect_process({key: actor[key] for key in ("pid", "create_time")}))
        for child in actor["children"]:
            # These are the only non-Ray processes this fixture may terminate.
            children.append(_inspect_process(child, terminate=terminate))
        endpoint = actor.get("endpoint")
        if endpoint:
            try:
                with socket.create_connection((endpoint["host"], endpoint["port"]), timeout=0.5):
                    endpoints.append({**endpoint, "closed": False})
            except OSError as exc:
                # Refusal confirms that there is no accepting listener here.
                # Timeout, unreachable host, and permission errors do not.
                if exc.errno != errno.ECONNREFUSED:
                    raise
                endpoints.append({**endpoint, "closed": True})
    return {"node_id": node_id, "actors": processes, "children": children, "endpoints": endpoints}


async def _observe_node(actors, terminate, timeout_s):
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    reference = ray.remote(_inspect_node).options(
        num_cpus=0,
        max_retries=0,
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=actors[0]["node_id"], soft=False),
    ).remote(actors, terminate)
    try:
        return await asyncio.wait_for(reference, timeout=timeout_s)
    except asyncio.CancelledError:
        ray.cancel(reference, force=True)
        raise
    except asyncio.TimeoutError:
        ray.cancel(reference, force=True)
        raise


def _ping_actor(actor):
    return True


async def _actor_dead(handle, timeout_s):
    from ray import exceptions

    try:
        await asyncio.wait_for(handle.__ray_call__.remote(_ping_actor), timeout=timeout_s)
    except asyncio.TimeoutError:
        return {"dead": False, "state": "RPC_TIMEOUT"}
    except exceptions.RayActorError as exc:
        unavailable = getattr(exceptions, "ActorUnavailableError", ())
        if isinstance(exc, unavailable):
            return {"dead": False, "state": "ACTOR_UNAVAILABLE"}
        return {"dead": True, "state": type(exc).__name__}
    return {"dead": False, "state": "RPC_ALIVE"}


async def cleanup_runtime(replica, snapshot: dict, timeout_s: float = 120) -> dict:
    """Kill owned actors and independently verify their processes/HTTP exit.

    After a short grace period, terminate only captured HTTP descendants whose
    PID and start time still match; escalate to kill for those same identities.
    Unknown RPC/node state and partial cleanup fail closed. Linux zombies have
    exited but may remain unreaped under container PID 1; that distinction is
    reported explicitly. This function never returns GS claim-release evidence
    and never restores donor engines.
    """
    if (
        isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float))
        or not math.isfinite(timeout_s) or timeout_s <= 0
    ):
        raise ValueError("cleanup timeout_s must be finite and positive")
    snapshot = copy.deepcopy(snapshot)
    handles = _validate_snapshot(replica, snapshot)
    started = time.monotonic()
    deadline = started + timeout_s
    diagnostics = {"replica_rank": replica.replica_rank, "release_confirmed": False, "errors": []}
    try:
        refreshed = await asyncio.wait_for(capture_runtime(replica), timeout=min(30.0, timeout_s / 2))
        snapshot = _merge_capture(snapshot, refreshed)
        _validate_snapshot(replica, snapshot)
        diagnostics["capture_refreshed"] = True
    except Exception as exc:
        # Still attempt teardown of the known owned runtime after a failed
        # refresh, but its incomplete process-tree evidence must never pass.
        diagnostics["capture_refreshed"] = False
        diagnostics["errors"].append({
            "phase": "refresh_capture", "type": type(exc).__name__, "message": str(exc) or repr(exc),
        })
    by_node = {}
    for actor in snapshot["actors"]:
        by_node.setdefault(actor["node_id"], []).append(actor)
    try:
        diagnostics["kill"] = await asyncio.wait_for(
            replica._cleanup_runtime(), timeout=max(0.001, deadline - time.monotonic())
        )
        diagnostics["errors"].extend(diagnostics["kill"].get("errors", []))
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            elapsed = time.monotonic() - started
            terminate = "kill" if elapsed >= 5 else "terminate" if elapsed >= 2 else None
            observations = await asyncio.gather(
                *(_actor_dead(handle, min(1.0, remaining)) for handle, _ in handles.values()),
                # A zero-CPU task may still need a cold Ray worker, including
                # accelerator imports; allow startup within the total budget.
                *(_observe_node(actors, terminate, min(30.0, remaining)) for actors in by_node.values()),
                return_exceptions=True,
            )
            failures = [item for item in observations if isinstance(item, BaseException)]
            if failures:
                diagnostics["errors"].extend(
                    {"type": type(exc).__name__, "message": str(exc) or repr(exc)} for exc in failures
                )
                break
            rpc_states = dict(zip(handles, observations[: len(handles)], strict=True))
            nodes = observations[len(handles) :]
            diagnostics.update({"actor_rpc": rpc_states, "nodes": nodes})
            actors_dead = all(item["dead"] for item in rpc_states.values()) and all(
                process["exited"] for node in nodes for process in node["actors"]
            )
            children_dead = all(child["exited"] for node in nodes for child in node["children"])
            endpoint_closed = all(endpoint["closed"] for node in nodes for endpoint in node["endpoints"])
            unreaped = [
                {"node_id": node["node_id"], **process}
                for node in nodes for process in node["actors"] + node["children"]
                if process["exited"] and not process["reaped"]
            ]
            diagnostics.update({
                "actors_dead": actors_dead,
                "child_processes_dead": children_dead,
                "endpoint_closed": endpoint_closed,
                "process_table_reaped": all(
                    process["reaped"] for node in nodes for process in node["actors"] + node["children"]
                ),
                "unreaped_processes": unreaped,
            })
            if actors_dead and children_dead and endpoint_closed:
                if diagnostics["errors"]:
                    break
                return {
                    **diagnostics,
                    "state": "TEST_RUNTIME_CLEANED",
                    "release_confirmed": True,
                    "released": False,
                    "elapsed_s": time.monotonic() - started,
                }
            await asyncio.sleep(min(0.2, max(0.0, deadline - time.monotonic())))
    except Exception as exc:
        diagnostics["errors"].append({"type": type(exc).__name__, "message": str(exc) or repr(exc)})
    if not diagnostics["errors"]:
        diagnostics["errors"].append("timed out waiting for actor, child-process, and endpoint release")
    diagnostics["state"] = "TEST_RUNTIME_CLEANUP_FAILED"
    raise RuntimeCleanupError(diagnostics)
