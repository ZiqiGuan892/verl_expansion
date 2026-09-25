"""Opt-in, idle-window generation checks against the existing real load balancer.

This fixture must run before the native fit loop starts or after both native
components finish. Its caller owns that boundary; an idle LB alone does not
prevent a live producer from submitting another request.
"""

import asyncio
from uuid import uuid4


async def probe_replicas(
    manager,
    tokenizer,
    replica_ranks: list[int],
    expected_version: int,
    concurrency: int = 1,
    *,
    timeout_seconds: float = 120,
    max_tokens: int = 16,
) -> dict:
    """Generate through each selected server and restore the original LB pool.

    No runtime or replacement balancer is created. The native client performs
    acquisition, generation, and release; this fixture observes acquisitions.
    Each RPC and each concurrent generation batch has a bounded wait. A timeout
    is a failed test, never proof that a remote engine request was cancelled.
    """
    if not replica_ranks or len(set(replica_ranks)) != len(replica_ranks):
        raise ValueError("replica_ranks must be nonempty and unique")
    if any(isinstance(rank, bool) or not isinstance(rank, int) or rank < 0 for rank in replica_ranks):
        raise ValueError("replica_ranks must contain non-negative integers")
    if isinstance(expected_version, bool) or not isinstance(expected_version, int) or expected_version < 0:
        raise ValueError("expected_version must be a non-negative integer")
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
        raise ValueError("concurrency must be a positive integer")
    if timeout_seconds <= 0 or isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError("timeout_seconds and max_tokens must be positive")

    # Keep importing this test module dependency-light. The actual probe fails
    # explicitly if the native client/runtime is unavailable.
    from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient

    class ObservedClient(FullyAsyncLLMServerClient):
        async def _acquire_server(self, request_id: str, **extra):
            server_id, handle = await super()._acquire_server(request_id, **extra)
            acquisitions.setdefault(request_id, []).append(server_id)
            return server_id, handle

    lb = manager.global_load_balancer

    async def rpc(method, **kwargs):
        return await asyncio.wait_for(method.remote(**kwargs), timeout=timeout_seconds)

    async def wait_idle():
        async def poll():
            while True:
                count = await lb.get_total_inflight.remote()
                if count == 0:
                    return 0
                await asyncio.sleep(0.05)

        return await asyncio.wait_for(poll(), timeout=timeout_seconds)

    inflight_before = await rpc(lb.get_total_inflight)
    if inflight_before != 0:
        raise RuntimeError(f"generation probe requires an idle LB, got inflight={inflight_before}")
    original_ids = list(await rpc(lb.get_all_servers))
    known_handles = dict(zip(manager.server_addresses, manager.server_handles, strict=True))
    missing = set(original_ids) - known_handles.keys()
    if missing:
        raise RuntimeError(f"cannot restore LB routes with unknown manager handles: {sorted(missing)}")
    targets = []
    for rank in replica_ranks:
        replica = next((item for item in manager.rollout_replicas if item.replica_rank == rank), None)
        if replica is None:
            record = manager._borrowed_record_by_rank(rank)
            replica = record.get("replica")
        server_id = getattr(replica, "_server_address", None)
        if server_id not in original_ids:
            raise RuntimeError(f"replica {rank} has no published primary server route")
        targets.append((rank, server_id))
    if len({server_id for _, server_id in targets}) != len(targets):
        raise RuntimeError("distinct replicas must have distinct primary servers")
    prompt_ids = list(tokenizer.encode("Write one short sentence about the sky.", add_special_tokens=False))
    if not prompt_ids:
        raise RuntimeError("generation probe tokenizer produced an empty prompt")

    acquisitions = {}
    client = manager.get_client(client_cls=ObservedClient)
    receipts = []
    try:
        for rank, server_id in targets:
            await wait_idle()
            current_ids = set(await rpc(lb.get_all_servers))
            if current_ids != set(original_ids):
                raise RuntimeError("LB routes changed outside the idle generation fixture")
            await rpc(lb.remove_servers, server_ids=sorted(current_ids - {server_id}))
            await rpc(lb.clear_sticky_cache)
            request_ids = [f"d4-e2e-{rank}-{uuid4().hex}" for _ in range(concurrency)]

            async def generate(request_id):
                output = await client.generate(
                    request_id=request_id,
                    prompt_ids=prompt_ids,
                    sampling_params={"max_tokens": max_tokens, "temperature": 0.0},
                )
                actual_servers = acquisitions.get(request_id, [])
                if not actual_servers or any(item != server_id for item in actual_servers):
                    raise RuntimeError(f"request {request_id} did not acquire selected server {server_id}")
                fields = output.extra_fields
                versions = [fields.get(key) for key in ("global_steps", "min_global_steps", "max_global_steps")]
                if any(
                    isinstance(version, bool) or not isinstance(version, int) or version != expected_version
                    for version in versions
                ):
                    raise RuntimeError(f"request {request_id} has versions {versions}, expected {expected_version}")
                tokens = list(output.token_ids)
                if not tokens or output.stop_reason in {None, "aborted", "abort"}:
                    raise RuntimeError(f"request {request_id} returned empty or aborted generation")
                return {
                    "request_id": request_id,
                    "server_id": server_id,
                    "acquired_server_ids": actual_servers,
                    "token_count": len(tokens),
                    "token_ids": tokens,
                    "global_steps": versions[0],
                    "min_version": versions[1],
                    "max_version": versions[2],
                    "stop_reason": output.stop_reason,
                }

            tasks = [asyncio.create_task(generate(request_id)) for request_id in request_ids]
            try:
                requests = await asyncio.wait_for(asyncio.gather(*tasks), timeout=timeout_seconds)
            finally:
                # gather does not cancel sibling requests when one raises.
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            await wait_idle()  # Native release is fire-and-forget.
            receipts.append({"replica_rank": rank, "server_id": server_id, "requests": requests})
            current_ids = set(await rpc(lb.get_all_servers))
            await rpc(
                lb.add_servers,
                servers={sid: known_handles[sid] for sid in original_ids if sid not in current_ids},
            )
    finally:
        # Only missing original routes are added: never reset a surviving
        # server's in-flight counter, including on generation failure.
        current_ids = set(await rpc(lb.get_all_servers))
        await rpc(lb.add_servers, servers={sid: known_handles[sid] for sid in original_ids if sid not in current_ids})
        await rpc(lb.clear_sticky_cache)
        if set(await rpc(lb.get_all_servers)) != set(original_ids):
            raise RuntimeError("generation fixture could not restore the original LB routes")
    inflight_after = await wait_idle()
    return {
        "state": "GENERATED",
        "version": expected_version,
        "concurrency": concurrency,
        "replicas": receipts,
        "inflight_before": inflight_before,
        "inflight_after": inflight_after,
        "routes_restored": True,
    }
