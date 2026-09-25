"""Opt-in real-device acceptance fixture; never a production lifecycle manager.

Native initialization and fit run unchanged. This fixture owns a quiet window
before/after fit, substitutes only GS placement input, and uses real actors,
engines, requests, optimizer updates and checkpoint transport throughout.
"""

import asyncio
import copy
import json
import time


def enabled(config) -> bool:
    return bool(config.get("multitask", {}).get("e2e_test", {}).get("enabled", False))


async def prepare(rollouter, scenario: str) -> dict:
    manager = rollouter.llm_server_manager
    if getattr(rollouter, "_e2e_context", None) is not None:
        raise RuntimeError("E2E fixture already owns a test window")
    if scenario == "shared_bundle":
        specs = await manager._build_shared_bundle_test_specs()
    else:
        placement = {
            "split": "basic", "cross_pg": "merge_world_size", "idempotent": "basic",
            "concurrent_idempotent": "basic", "pressure": "basic",
        }.get(scenario, scenario)
        spec, negative = await manager._build_d2_test_spec(placement)
        if negative:
            raise ValueError("negative scenarios have a separate acceptance entry")
        specs = [spec]
        if scenario == "split":
            if spec["world_size"] != 4:
                raise ValueError("S2 requires exactly one four-device donor")
            specs = []
            for index in range(2):
                part = copy.deepcopy(spec)
                part["claims"] = manager._reindex_test_claims(part["claims"][index * 2 : index * 2 + 2])
                part["world_size"] = 2
                part["parallelism"]["tensor_model_parallel_size"] = 2
                for field in ("operation_id", "lease_id", "borrower_replica_id"):
                    part[field] += f"-part-{index}"
                specs.append(part)
    for spec in specs:
        # Test authorization spans startup + the actual training window.
        # Production GS controls its own lease expiry and is never bypassed.
        spec["expires_at"] = time.time() + 3600
    donors = {}
    for spec in specs:
        for donor in manager._local_native_donors(spec):
            donors[donor.replica_rank] = donor
    donor_sizes = [donor.world_size for donor in donors.values()]
    if scenario in {"cross_pg", "merge_world_size"}:
        if donor_sizes != [2, 2] or specs[0]["world_size"] != 4:
            raise ValueError("merge acceptance requires exactly two TP=2 donors and one TP=4 borrower")
        if len({claim["pg_id"] for claim in specs[0]["claims"]}) != 2:
            raise ValueError("merge acceptance must span two distinct native PGs")
    native = list(manager.rollout_replicas)
    if any(getattr(replica, "allocation_kind", "native") != "native" for replica in native):
        raise RuntimeError("E2E fixture must start before any borrowed runtime is created")
    rollouter._e2e_context = {
        "scenario": scenario, "specs": specs, "donors": list(donors.values()),
        "native": native, "routes": dict(zip(manager.server_addresses, manager.server_handles, strict=True)),
        "slept": [], "snapshots": {}, "cleaned": [],
    }
    return {"specs": specs, "donor_ranks": list(donors), "donor_world_sizes": donor_sizes}


async def _quiet_lb(manager, timeout=60):
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("E2E window still has in-flight requests")
        if not await asyncio.wait_for(manager.global_load_balancer.get_total_inflight.remote(), remaining):
            return
        await asyncio.sleep(0.2)


async def dispatch_rollout(rollouter, action: str, payload: dict) -> dict:
    """All mutable handles stay inside the owning Rollouter/manager process."""
    from .e2e_cleanup import capture_runtime, cleanup_runtime
    from .e2e_generation import probe_replicas

    manager = rollouter.llm_server_manager
    if action == "prepare":
        return await prepare(rollouter, payload["scenario"])
    context = rollouter._e2e_context
    if action == "owned":
        # A create RPC can raise *after* creating its runtime (for example in
        # bootstrap). The manager's lease records, not successful RPC replies,
        # are the authoritative inventory of this fixture's side effects.
        return {"replica_ranks": [
            manager.borrowed_operations[spec["lease_id"]]["replica_rank"]
            for spec in context["specs"] if spec["lease_id"] in manager.borrowed_operations
        ]}
    if action == "park":
        await _quiet_lb(manager)
        await manager.global_load_balancer.remove_servers.remote(list(context["routes"]))
        await manager.global_load_balancer.clear_sticky_cache.remote()
        for donor in context["donors"]:
            await manager._test_memory_call(donor, "sleep_for_runtime_test")
            context["slept"].append(donor)
        return {"state": "TEST_DONORS_PARKED"}
    if action == "capture":
        rank = payload["replica_rank"]
        replica = manager._borrowed_record_by_rank(rank)["replica"]
        context["snapshots"][rank] = await capture_runtime(replica)
        return {"replica_rank": rank, "captured": True}
    if action == "probe":
        return await probe_replicas(
            manager, rollouter.tokenizer, payload["replica_ranks"], payload["version"],
            concurrency=payload.get("concurrency", 1),
        )
    if action == "audit":
        await _quiet_lb(manager)
        results = []
        for rank in payload["replica_ranks"]:
            replica = manager._borrowed_record_by_rank(rank)["replica"]
            if payload.get("reset", False):
                result = await replica.servers[0].set_test_generation_audit.remote(True, True)
            else:
                result = await replica.servers[0].get_test_generation_audit.remote()
            results.append({"replica_rank": rank, **result, "completed": result.get("nonempty_calls", 0)})
        return {"replicas": results}
    if action in {"pause", "wake"}:
        rank = payload["replica_rank"]
        record = manager._borrowed_record_by_rank(rank)
        replica = record["replica"]
        if action == "pause":
            await _quiet_lb(manager)
            address = replica._server_address
            await manager.global_load_balancer.remove_servers.remote([address])
            if address in manager.server_addresses:
                index = manager.server_addresses.index(address)
                manager.server_addresses.pop(index)
                manager.server_handles.pop(index)
            manager.ready_replica_ranks.discard(rank)
            record["state"] = "RUNTIME_READY"
            await manager._test_memory_call(replica, "sleep_for_runtime_test")
        else:
            await manager._test_memory_call(replica, "wake_for_runtime_test")
        return {"replica_rank": rank, "state": f"TEST_{action.upper()}"}
    if action == "topology":
        ranks = payload["replica_ranks"]
        replicas = [manager._borrowed_record_by_rank(rank)["replica"] for rank in ranks]
        sizes = [replica.world_size for replica in replicas]
        if sizes != [spec["world_size"] for spec in context["specs"]]:
            raise RuntimeError("actual borrowed world sizes differ from the requested topology")
        devices = [
            {(item["node_id"], str(item["accelerator_id"])) for item in replica.actual_device_map.values()}
            for replica in replicas
        ]
        if len(set(ranks)) != len(ranks) or len({replica._server_address for replica in replicas}) != len(ranks):
            raise RuntimeError("borrowed ranks/endpoints must be independent")
        shared = context["scenario"] == "shared_bundle"
        if shared:
            if sizes != [1, 1] or devices[0] != devices[1] or len(devices[0]) != 1:
                raise RuntimeError("S5 did not create two runtimes on the same device")
        elif len(set.union(*devices)) != sum(sizes):
            raise RuntimeError("non-shared topology placed multiple ranks on the same device")
        for spec, actual in zip(context["specs"], devices, strict=True):
            expected = {(claim["node_id"], str(claim["accelerator_id"])) for claim in spec["claims"]}
            if expected != actual:
                raise RuntimeError("borrowed physical placement differs from the donor claims")
        return {
            "validated": True, "kind": context["scenario"], "replica_ranks": ranks,
            "world_sizes": sizes, "donor_world_sizes": [replica.world_size for replica in context["donors"]],
            "donor_ranks": [replica.replica_rank for replica in context["donors"]],
            "pg_ids": sorted({claim["pg_id"] for spec in context["specs"] for claim in spec["claims"]}),
            "shared_bundle": shared,
            "training_replica_ranks": [ranks[-1]] if shared else ranks,
        }
    if action == "cleanup":
        import ray

        await _quiet_lb(manager)
        records = [manager.borrowed_operations[spec["lease_id"]] for spec in context["specs"]
                   if spec["lease_id"] in manager.borrowed_operations]
        addresses = [record["replica"]._server_address for record in records
                     if record.get("replica") is not None and record["replica"]._server_address]
        await manager.global_load_balancer.remove_servers.remote(addresses)
        errors = []
        for record in records:
            rank = record["replica_rank"]
            if rank in {item["replica_rank"] for item in context["cleaned"]}:
                continue
            try:
                replica = record.get("replica")
                if replica is None:
                    raise RuntimeError("partial create has no inspectable runtime; release cannot be confirmed")
                if rank not in context["snapshots"]:
                    try:
                        context["snapshots"][rank] = await capture_runtime(replica)
                    except Exception:
                        # Known owned handles may still be killed, but without
                        # a complete process snapshot release stays unconfirmed.
                        await asyncio.wait_for(replica._cleanup_runtime(), timeout=30)
                        raise
                result = await cleanup_runtime(replica, context["snapshots"][rank])
                if not result.get("release_confirmed", False):
                    raise RuntimeError(f"borrowed resources not confirmed released: {result}")
                context["cleaned"].append({"replica_rank": rank, **result})
                manager.ready_replica_ranks.discard(rank)
                manager.rollout_replicas = [item for item in manager.rollout_replicas if item is not replica]
                record["state"] = "TEST_CLEANED"
            except Exception as error:
                # Attempt all fixture-owned borrowers even if one is uncertain.
                errors.append({"replica_rank": rank, "error": repr(error)})
        if errors:
            raise RuntimeError(f"E2E resources not fully released; donors remain parked: {errors}")
        remaining = set(await manager.global_load_balancer.get_all_servers.remote())
        if remaining.intersection(addresses):
            raise RuntimeError("borrowed routes survived cleanup")
        for spec in context["specs"]:
            for claim in spec["claims"]:
                pg = ray.util.get_placement_group(claim["pg_name"])
                if pg.id.hex() != claim["pg_id"] or ray.util.placement_group_table(pg).get("state") != "CREATED":
                    raise RuntimeError("donor PG was changed or removed")
        for donor in context["donors"]:
            await donor.servers[0].get_server_address.remote()
            await manager._snapshot_native_claims(donor, "e2e-native-owner-check")
        return {"test_resources_released": True, "lb_removed": True, "donor_pg_preserved": True,
                "borrowers": context["cleaned"]}
    if action == "wake_donors":
        for donor in context["slept"]:
            await manager._test_memory_call(donor, "wake_for_runtime_test")
        return {"state": "TEST_DONORS_AWAKE_NOT_ROUTED"}
    if action == "restore_routes":
        manager.server_addresses = list(context["routes"])
        manager.server_handles = list(context["routes"].values())
        await manager.global_load_balancer.add_servers.remote(context["routes"])
        await manager.global_load_balancer.clear_sticky_cache.remote()
        return {"state": "TEST_NATIVE_ROUTES_RESTORED"}
    raise ValueError(f"unknown E2E Rollouter action: {action}")


async def record_normal_sync(trainer, origin="optimizer_loop") -> dict:
    """Called inside the Trainer snapshot gate after a real CE transaction."""
    manager = trainer.checkpoint_manager
    if manager.sync_state != "IDLE":
        raise RuntimeError("ordinary sync has not finalized")
    replicas = list(manager._effective_replicas_unlocked())
    version = int(trainer.current_param_version)
    ranks = [replica.replica_rank for replica in replicas]
    versions = {str(rank): manager.last_synced_versions.get(rank) for rank in ranks}
    if not ranks or any(value != version for value in versions.values()):
        raise RuntimeError(f"ordinary sync did not confirm every effective replica: {versions}")
    validation = await manager.validate_parameter_sync(
        replicas, version, source_manifest=await manager._get_source_manifest()
    )
    for replica in replicas:
        if getattr(replica, "allocation_kind", "native") == "borrowed":
            await trainer.rollouter.mark_replica_serving_version.remote(replica.replica_rank, version)
    result = {"version": version, "origin": origin, "replica_ranks": ranks,
              "synchronized_versions": versions,
              "replica_worker_counts": {str(replica.replica_rank): len(replica.workers) for replica in replicas},
              "parameter_validation": validation}
    trainer._e2e_syncs.append(result)
    print(f"D0_D4_E2E_NORMAL_SYNC {json.dumps(result, sort_keys=True)}", flush=True)
    return result


async def dispatch_trainer(trainer, action: str, payload: dict) -> dict:
    if action == "enable":
        if not trainer.parameter_validation_enabled or not trainer.source_validation_enabled:
            raise RuntimeError("E2E requires actor-source and CE-receiver manifest validation")
        trainer._e2e_syncs = []
        return {"version": int(trainer.current_param_version)}
    if action == "state":
        return {"version": int(trainer.current_param_version), "normal_syncs": copy.deepcopy(trainer._e2e_syncs),
                "training": trainer.get_training_completion()}
    if action in {"sync_current", "restore_donors"}:
        async with trainer.parameter_snapshot_gate:
            if action == "restore_donors":
                # Test-only quiet window: routes remain absent until this full
                # native-only sync completes. Never label old donor weights.
                await trainer.checkpoint_manager.resume_replicas_for_sync(payload["replica_ranks"])
            await trainer.checkpoint_manager.update_weights(global_steps=trainer.current_param_version)
            return await record_normal_sync(trainer, origin="test_explicit_normal_sync")
    if action == "assert_removed":
        known = {replica.replica_rank for replica in trainer.checkpoint_manager.replicas}
        if known.intersection(payload["replica_ranks"]):
            raise RuntimeError("borrowed CE membership survived test unregister")
        return {"ce_unregistered": True}
    if action == "unregister":
        return await trainer.checkpoint_manager.unregister_replica(payload["replica_rank"])
    raise ValueError(f"unknown E2E Trainer action: {action}")


def run_training_fixture(runner, native_fit) -> None:
    """Wrap the inherited fit loop; no optimizer/version/sample is fabricated."""
    import ray
    from concurrent.futures import ThreadPoolExecutor

    config = runner.components["config"]
    scenario = config.get("multitask", {}).get("e2e_test", {}).get("scenario", "basic")
    trainer = runner.components["trainer"]
    rollouter = runner.components["rollouter"]

    def rollout(action, **payload):
        return ray.get(rollouter.e2e_test_action.remote(action, payload))

    def train(action, **payload):
        return ray.get(trainer.e2e_test_action.remote(action, payload))

    prepared = rollout("prepare", scenario=scenario)
    specs = prepared["specs"]
    donors = prepared["donor_ranks"]
    initial = train("enable")["version"]
    ranks, bootstrap_versions, before_parts = [], {}, []
    suspended = False
    cleaned = False
    shared = scenario == "shared_bundle"
    concurrency = 4 if scenario == "pressure" else 1
    idempotency = None
    try:
        ray.get(trainer.suspend_donors_for_borrow.remote(donors))
        suspended = True
        rollout("park")
        for spec in specs:
            if scenario == "concurrent_idempotent":
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures = [pool.submit(runner.execute_replica_operation, "create", copy.deepcopy(spec)) for _ in range(2)]
                    receipts = [future.result() for future in futures]
            else:
                receipts = [runner.execute_replica_operation("create", spec)]
                if scenario == "idempotent":
                    receipts.append(runner.execute_replica_operation("create", copy.deepcopy(spec)))
            result = receipts[0]
            rank = result.get("replica_rank")
            if rank is not None:
                ranks.append(int(rank))
            if any(receipt.get("state") != "LB_READY" for receipt in receipts):
                raise RuntimeError(f"E2E create did not reach LB_READY: {receipts}")
            if len(receipts) == 2:
                snapshot = ray.get(rollouter.test_operation_snapshot.remote(spec["lease_id"]))
                endpoints = {item.get("server_id") or item.get("ready", {}).get("server_id") for item in receipts}
                idempotency = {
                    "same_rank": all(item.get("replica_rank") == rank for item in receipts),
                    "same_server": endpoints == {snapshot.get("server_id")} and None not in endpoints,
                    "one_runtime": snapshot["worker_count"] == spec["world_size"]
                    and snapshot["server_count"] == len({claim["node_id"] for claim in spec["claims"]}),
                }
                if not all(idempotency.values()):
                    raise RuntimeError(f"E2E idempotency failed: {idempotency}")
            confirmed = [item["bootstrap"]["version"] for item in receipts if "bootstrap" in item]
            if not confirmed or any(version != initial for version in confirmed):
                raise RuntimeError(f"bootstrap did not confirm initial actor version {initial}: {result}")
            bootstrap_versions[str(rank)] = confirmed[0]
            rollout("capture", replica_rank=rank)
            before_parts.append(rollout("probe", replica_ranks=[rank], version=initial, concurrency=concurrency))
            if shared and len(ranks) == 1:
                train("sync_current")
                ray.get(trainer.unregister_replica.remote(rank))
                rollout("pause", replica_rank=rank)
        topology = rollout("topology", replica_ranks=ranks)
        if idempotency is not None:
            topology["idempotency"] = idempotency
        active = topology["training_replica_ranks"]
        rollout("audit", replica_ranks=active, reset=True)
        native_fit()
        state = train("state")
        if not state["training"].get("completed") or state["version"] <= initial:
            raise RuntimeError("native training did not finish with a newer real parameter version")
        audits = rollout("audit", replica_ranks=active)["replicas"]
        after_parts = [rollout("probe", replica_ranks=active, version=state["version"], concurrency=concurrency)]
        if shared:
            ray.get(trainer.unregister_replica.remote(ranks[1]))
            rollout("pause", replica_rank=ranks[1])
            rollout("wake", replica_rank=ranks[0])
            ray.get(trainer.register_replica.remote(ranks[0]))
            ray.get(trainer.bootstrap_replica.remote(ranks[0]))
            ray.get(rollouter.commit_replica_ready.remote(ranks[0]))
            after_parts.append(rollout("probe", replica_ranks=[ranks[0]], version=state["version"]))
            topology["activation_order"] = [ranks[0], ranks[1], ranks[0]]
        for rank in ranks:
            ray.get(trainer.unregister_replica.remote(rank))
        membership = train("assert_removed", replica_ranks=ranks)
        cleanup = rollout("cleanup", replica_ranks=ranks)
        cleaned = True
        rollout("wake_donors")
        restored = train("restore_donors", replica_ranks=donors)
        suspended = False
        rollout("restore_routes")
        donor_probe = rollout("probe", replica_ranks=donors, version=state["version"])
        cleanup.update(membership | {"donor_restored": True, "donor_sync_version": restored["version"],
                                     "donor_parameter_validation": restored["parameter_validation"],
                                     "donor_generation": donor_probe})
        def combine(parts, version):
            if not parts or any(part.get("state") != "GENERATED" or part.get("version") != version
                                or part.get("inflight_before") != 0 or part.get("inflight_after") != 0
                                or part.get("routes_restored") is not True for part in parts):
                raise RuntimeError("generation evidence is incomplete or its LB counters did not drain")
            return {"state": "GENERATED", "version": version,
                    "concurrency": min(part["concurrency"] for part in parts),
                    "inflight_before": 0, "inflight_after": 0, "routes_restored": True,
                    "replicas": [item for part in parts for item in part["replicas"]]}

        result = {"schema_version": 1, "scenario": scenario, "state": "PASSED",
                  "training": state["training"], "topology": topology, "bootstrap_versions": bootstrap_versions,
                  "normal_syncs": state["normal_syncs"], "training_audits": audits,
                  "generation_before": combine(before_parts, initial),
                  "generation_after": combine(after_parts, state["version"]), "cleanup": cleanup}
        from .e2e_verdict import validate_result

        validate_result(result, scenario)
        print(f"D0_D4_E2E_RESULT {json.dumps(result, sort_keys=True)}", flush=True)
    except Exception:
        # Test-owned teardown only. A failed release never wakes donors into
        # unknown memory or produces a success receipt.
        if not cleaned:
            try:
                owned_ranks = rollout("owned")["replica_ranks"]
                for rank in owned_ranks:
                    train("unregister", replica_rank=rank)
                train("assert_removed", replica_ranks=owned_ranks)
                rollout("cleanup", replica_ranks=owned_ranks)
                cleaned = True
            except Exception as cleanup_error:
                print(f"D0_D4_E2E_CLEANUP_ERROR {cleanup_error!r}", flush=True)
        if cleaned and suspended:
            try:
                rollout("wake_donors")
                train("restore_donors", replica_ranks=donors)
                rollout("restore_routes")
            except Exception as restore_error:
                print(f"D0_D4_E2E_RESTORE_ERROR {restore_error!r}", flush=True)
        raise
