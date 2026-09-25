"""Validate one process's complete E2E receipt without importing Ray or verl.

The caller must supply the actual process exit status. Independent log markers
are deliberately not combined: all evidence belongs to one final JSON receipt.
This checks recorded evidence; only the real NPU run can produce acceptance.
"""

import argparse
import json
from pathlib import Path
import sys


RESULT_MARKER = "D0_D4_E2E_RESULT "
SCENARIOS = {
    "basic", "split", "cross_pg", "fragmented", "shared_bundle",
    "merge_world_size", "idempotent", "concurrent_idempotent", "pressure",
}


class E2EVerdictError(ValueError):
    """The process or its structured evidence did not satisfy acceptance."""


def _require(condition, message):
    if not condition:
        raise E2EVerdictError(message)


def _mapping(value, name):
    _require(isinstance(value, dict), f"{name} must be an object")
    return value


def _sequence(value, name):
    _require(isinstance(value, list) and bool(value), f"{name} must be a nonempty list")
    return value


def _integer(value, name, minimum=0):
    _require(type(value) is int and value >= minimum, f"{name} must be an integer >= {minimum}")
    return value


def _ranks(value, name):
    values = [_integer(item, name) for item in _sequence(value, name)]
    _require(len(set(values)) == len(values), f"{name} contains duplicate ranks")
    return values


def _text(value, name):
    _require(isinstance(value, str) and bool(value.strip()), f"{name} must be a nonempty string")
    return value


def _rank_mapping(value, ranks, name):
    result = _mapping(value, name)
    _require(set(result) == {str(rank) for rank in ranks}, f"{name} must cover exactly its replica ranks")
    return result


def _generation(value, ranks, version, phase, request_ids, pressure):
    receipt = _mapping(value, phase)
    _require(receipt.get("state") == "GENERATED", f"{phase} did not generate")
    _require(_integer(receipt.get("version"), f"{phase}.version") == version, f"{phase} version mismatch")
    for key in ("inflight_before", "inflight_after"):
        _require(_integer(receipt.get(key), f"{phase}.{key}") == 0, f"{phase} left in-flight requests")
    _require(receipt.get("routes_restored") is True, f"{phase} did not restore routes")
    replicas = _sequence(receipt.get("replicas"), f"{phase}.replicas")
    seen_ranks = set()
    servers = {}
    if pressure:
        _integer(receipt.get("concurrency"), f"{phase}.concurrency", minimum=4)
    for item in replicas:
        replica = _mapping(item, f"{phase}.replica")
        rank = _integer(replica.get("replica_rank"), f"{phase}.replica_rank")
        _require(rank in ranks and rank not in seen_ranks, f"{phase} has unexpected or duplicate rank {rank}")
        seen_ranks.add(rank)
        server = _text(replica.get("server_id"), f"{phase}.server_id")
        _require(server not in servers.values(), f"{phase} reuses a server for different ranks")
        servers[rank] = server
        requests = _sequence(replica.get("requests"), f"{phase}.requests")
        _require(not pressure or len(requests) >= 4, f"{phase} pressure requires >=4 requests per replica")
        for entry in requests:
            request = _mapping(entry, f"{phase}.request")
            request_id = _text(request.get("request_id"), f"{phase}.request_id")
            _require(request_id not in request_ids, "generation request_id was reused")
            request_ids.add(request_id)
            _require(request.get("server_id") == server, f"{phase} request server mismatch")
            acquired = _sequence(request.get("acquired_server_ids"), f"{phase}.acquired_server_ids")
            _require(all(item == server for item in acquired), f"{phase} acquired a different server")
            count = _integer(request.get("token_count"), f"{phase}.token_count", minimum=1)
            tokens = _sequence(request.get("token_ids"), f"{phase}.token_ids")
            _require(len(tokens) == count, f"{phase} token count differs from returned tokens")
            for token in tokens:
                _integer(token, f"{phase}.token_id")
            for key in ("global_steps", "min_version", "max_version"):
                _require(_integer(request.get(key), f"{phase}.{key}") == version, f"{phase} request {key} mismatch")
            reason = _text(request.get("stop_reason"), f"{phase}.stop_reason")
            _require(reason.lower() not in {"abort", "aborted", "error", "failed"}, f"{phase} generation aborted")
    _require(seen_ranks == set(ranks), f"{phase} does not cover every borrowed rank")
    return servers


def validate_result(result, expected_scenario):
    """Validate the internally linked evidence from one parsed final receipt."""
    _require(expected_scenario in SCENARIOS, f"unsupported scenario: {expected_scenario}")
    result = _mapping(result, "result")
    _require(type(result.get("schema_version")) is int and result["schema_version"] == 1, "unsupported schema_version")
    _require(result.get("scenario") == expected_scenario, "scenario mismatch")
    _require(result.get("state") == "PASSED", "E2E result did not pass")

    training = _mapping(result.get("training"), "training")
    _require(training.get("completed") is True and training.get("state") == "COMPLETED", "training did not complete")
    target = _integer(training.get("target_steps"), "training.target_steps", minimum=1)
    _require(_integer(training.get("completed_steps"), "training.completed_steps", minimum=1) == target,
             "training did not complete exactly its target steps")
    final_version = _integer(training.get("current_param_version"), "training.current_param_version", minimum=1)

    topology = _mapping(result.get("topology"), "topology")
    _require(topology.get("validated") is True and topology.get("kind") == expected_scenario, "topology was not validated for scenario")
    ranks = _ranks(topology.get("replica_ranks"), "topology.replica_ranks")
    worlds = [_integer(item, "topology.world_sizes", minimum=1)
              for item in _sequence(topology.get("world_sizes"), "topology.world_sizes")]
    _require(len(worlds) == len(ranks), "world_sizes must correspond to replica_ranks")
    world_by_rank = dict(zip(ranks, worlds))
    donors = [_integer(item, "topology.donor_world_sizes", minimum=1)
              for item in _sequence(topology.get("donor_world_sizes"), "topology.donor_world_sizes")]
    donor_ranks = _ranks(topology.get("donor_ranks"), "topology.donor_ranks")
    _require(len(donor_ranks) == len(donors) and not set(donor_ranks).intersection(ranks),
             "donor and borrowed rank ownership is inconsistent")
    pgs = [_text(item, "topology.pg_ids") for item in _sequence(topology.get("pg_ids"), "topology.pg_ids")]
    _require(len(set(pgs)) == len(pgs), "topology.pg_ids must be unique")
    active = _ranks(topology.get("training_replica_ranks", ranks), "topology.training_replica_ranks")
    _require(set(active) <= set(ranks), "training contains a rank outside the borrowed topology")
    if expected_scenario == "split":
        _require(worlds == [2, 2] and donors == [4] and len(pgs) == 1,
                 "split must create two world_size=2 replicas from one world_size=4 donor/PG")
    if expected_scenario in {"cross_pg", "merge_world_size"}:
        _require(worlds == [4] and donors == [2, 2] and len(pgs) == 2,
                 "cross-PG/merge requires one world_size=4 borrower from two world_size=2 donors/PGs")
    if expected_scenario == "shared_bundle":
        _require(worlds == [1, 1] and len(pgs) == 1 and topology.get("shared_bundle") is True,
                 "shared_bundle requires two world_size=1 replicas on one shared bundle/PG")
        _require(active == [ranks[1]] and topology.get("activation_order") == [ranks[0], ranks[1], ranks[0]],
                 "shared_bundle must train B between A activation and A restoration")
    else:
        _require(set(active) == set(ranks), "every borrowed replica must participate in training")
        if expected_scenario != "split":
            _require(len(ranks) == 1, f"{expected_scenario} requires exactly one borrowed replica")
    if expected_scenario in {"idempotent", "concurrent_idempotent"}:
        idempotency = _mapping(topology.get("idempotency"), "topology.idempotency")
        _require(len(ranks) == 1 and all(idempotency.get(key) is True for key in ("same_rank", "same_server", "one_runtime")),
                 "idempotent create did not preserve one rank/server/runtime")

    bootstrap = _rank_mapping(result.get("bootstrap_versions"), ranks, "bootstrap_versions")
    bootstrap_values = [_integer(bootstrap[str(rank)], "bootstrap_versions.version") for rank in ranks]
    _require(len(set(bootstrap_values)) == 1, "before-generation requires one common bootstrap version")
    before_version = bootstrap_values[0]
    _require(final_version > before_version, "training did not advance beyond bootstrap")

    covered = set()
    last_optimizer_version = -1
    for value in _sequence(result.get("normal_syncs"), "normal_syncs"):
        sync = _mapping(value, "normal_sync")
        version = _integer(sync.get("version"), "normal_sync.version")
        _require(version <= final_version, "normal sync version is newer than completed training")
        sync_ranks = _ranks(sync.get("replica_ranks"), "normal_sync.replica_ranks")
        versions = _rank_mapping(sync.get("synchronized_versions"), sync_ranks, "normal_sync.synchronized_versions")
        workers = _rank_mapping(sync.get("replica_worker_counts"), sync_ranks, "normal_sync.replica_worker_counts")
        for rank in sync_ranks:
            _require(_integer(versions[str(rank)], "normal_sync.synchronized_version") == version, "normal sync rank version mismatch")
            count = _integer(workers[str(rank)], "normal_sync.worker_count", minimum=1)
            if rank in world_by_rank:
                _require(count == world_by_rank[rank], "normal sync omitted borrowed workers")
        validation = _mapping(sync.get("parameter_validation"), "normal_sync.parameter_validation")
        _require(validation.get("state") == "PARAMETERS_VALIDATED" and
                 validation.get("source_state") == "SOURCE_TO_RECEIVER_VALIDATED", "source-to-receiver parameter validation missing")
        _require(_integer(validation.get("version"), "parameter_validation.version") == version, "parameter validation version mismatch")
        _require(_integer(validation.get("worker_count"), "parameter_validation.worker_count", minimum=1) == sum(workers.values()),
                 "parameter validation did not cover the effective worker set")
        _integer(validation.get("parameter_count"), "parameter_validation.parameter_count", minimum=1)
        _integer(validation.get("total_numel"), "parameter_validation.total_numel", minimum=1)
        digest = _text(validation.get("manifest_digest"), "parameter_validation.manifest_digest")
        _require(_text(validation.get("source_manifest_digest"), "parameter_validation.source_manifest_digest") == digest,
                 "actor source and receiver manifests differ")
        origin = sync.get("origin")
        _require(origin in {"optimizer_loop", "test_explicit_normal_sync"}, "normal sync origin is missing or unknown")
        if origin == "optimizer_loop":
            _require(version >= last_optimizer_version, "optimizer sync versions moved backwards")
            last_optimizer_version = version
            if version == final_version and version > before_version:
                covered.update(set(sync_ranks) & set(active))
    _require(covered == set(active), "final optimizer-loop normal sync did not cover every training replica")

    request_ids = set()
    before_servers = _generation(result.get("generation_before"), ranks, before_version,
                                 "generation_before", request_ids, expected_scenario == "pressure")
    after_servers = _generation(result.get("generation_after"), ranks, final_version,
                                "generation_after", request_ids, expected_scenario == "pressure")
    _require(before_servers == after_servers, "borrowed servers changed between before/after generation")
    audits = _sequence(result.get("training_audits"), "training_audits")
    audited = set()
    for value in audits:
        audit = _mapping(value, "training_audit")
        rank = _integer(audit.get("replica_rank"), "training_audit.replica_rank")
        _require(rank in active and rank not in audited, "training audit has duplicate or unexpected rank")
        audited.add(rank)
        _integer(audit.get("completed"), "training_audit.completed", minimum=1)
        _integer(audit.get("token_count"), "training_audit.token_count", minimum=1)
        for key in ("failed_calls", "inflight"):
            _require(_integer(audit.get(key), f"training_audit.{key}") == 0,
                     f"training audit has {key}")
    _require(audited == set(active), "training audit omitted a training replica")

    cleanup = _mapping(result.get("cleanup"), "cleanup")
    for key in ("test_resources_released", "ce_unregistered", "lb_removed", "donor_pg_preserved", "donor_restored"):
        _require(cleanup.get(key) is True, f"cleanup.{key} was not confirmed")
    _require(_integer(cleanup.get("donor_sync_version"), "cleanup.donor_sync_version") == final_version,
             "donor was not restored to the completed training version")
    released = set()
    for receipt in _sequence(cleanup.get("borrowers"), "cleanup.borrowers"):
        item = _mapping(receipt, "cleanup.borrower")
        rank = _integer(item.get("replica_rank"), "cleanup.borrower.replica_rank")
        _require(rank in ranks and rank not in released, "cleanup borrower ownership is inconsistent")
        released.add(rank)
        _require(item.get("state") == "TEST_RUNTIME_CLEANED" and item.get("errors") == [],
                 "borrower cleanup failed or contains errors")
        for key in ("release_confirmed", "actors_dead", "child_processes_dead", "endpoint_closed"):
            _require(item.get(key) is True, f"cleanup borrower {rank} has no {key} evidence")
    _require(released == set(ranks), "cleanup omitted borrowed runtimes")
    donor_validation = _mapping(cleanup.get("donor_parameter_validation"), "cleanup.donor_parameter_validation")
    _require(donor_validation.get("state") == "PARAMETERS_VALIDATED" and
             donor_validation.get("source_state") == "SOURCE_TO_RECEIVER_VALIDATED" and
             donor_validation.get("version") == final_version,
             "donor restored weights were not validated against current actor source")
    _require(_integer(donor_validation.get("worker_count"), "donor_parameter_validation.worker_count", minimum=1)
             >= sum(donors), "donor sync omitted workers")
    digest = _text(donor_validation.get("manifest_digest"), "donor_parameter_validation.manifest_digest")
    _require(donor_validation.get("source_manifest_digest") == digest, "donor source manifest differs")
    _generation(cleanup.get("donor_generation"), donor_ranks, final_version,
                "donor_generation", request_ids, False)
    return result


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def validate_log(log_text, expected_scenario, process_exit_code):
    """Require process success and exactly one complete, parseable final receipt."""
    _require(type(process_exit_code) is int and process_exit_code == 0, "main_ppo process exit status was not zero")
    _require(log_text.count(RESULT_MARKER) == 1, "expected exactly one E2E result marker")
    line = next(line for line in log_text.splitlines() if RESULT_MARKER in line)
    payload = line.split(RESULT_MARKER, 1)[1]
    try:
        result = json.loads(payload, object_pairs_hook=_unique_json_object)
    except (json.JSONDecodeError, ValueError) as error:
        raise E2EVerdictError(f"invalid E2E result JSON: {error}") from error
    return validate_result(result, expected_scenario)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_file", type=Path)
    parser.add_argument("expected_scenario", choices=sorted(SCENARIOS))
    parser.add_argument("--process-exit-code", required=True, type=int)
    args = parser.parse_args(argv)
    try:
        validate_log(args.log_file.read_text(encoding="utf-8"), args.expected_scenario, args.process_exit_code)
    except (OSError, UnicodeError, E2EVerdictError) as error:
        print(f"E2E FAIL: {error}", file=sys.stderr)
        return 1
    print(f"E2E PASS: {args.expected_scenario}; validated complete receipt in {args.log_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
