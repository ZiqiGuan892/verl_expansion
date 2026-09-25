"""Synthetic receipt validation tests; these do not run Ray or NPU inference."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "src/multi_task_scheduler/testing/e2e_verdict.py"
spec = importlib.util.spec_from_file_location("e2e_verdict_under_test", MODULE_PATH)
verdict = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verdict)


def _receipt(scenario="basic"):
    ranks = [3, 4] if scenario in {"split", "shared_bundle"} else [3]
    worlds = [2, 2] if scenario == "split" else [4] * len(ranks)
    if scenario == "shared_bundle":
        worlds = [1, 1]
    active = ranks[-1:] if scenario == "shared_bundle" else ranks
    cross = scenario in {"cross_pg", "merge_world_size"}
    concurrency = 4 if scenario == "pressure" else 1
    topology = {
        "kind": scenario, "validated": True, "replica_ranks": ranks,
        "world_sizes": worlds, "donor_world_sizes": [2, 2] if cross else [4],
        "donor_ranks": [0, 1] if cross else [0],
        "pg_ids": ["pg-a", "pg-b"] if cross else ["pg-a"], "training_replica_ranks": active,
    }
    if scenario == "shared_bundle":
        topology.update(shared_bundle=True, activation_order=[3, 4, 3])
    if scenario in {"idempotent", "concurrent_idempotent"}:
        topology["idempotency"] = {"same_rank": True, "same_server": True, "one_runtime": True}

    def generation(version, selected_ranks=None):
        return {
            "state": "GENERATED", "version": version, "concurrency": concurrency,
            "inflight_before": 0, "inflight_after": 0, "routes_restored": True,
            "replicas": [{
                "replica_rank": rank, "server_id": f"server-{rank}",
                "requests": [{
                    "request_id": f"req-{version}-{rank}-{index}", "server_id": f"server-{rank}",
                    "acquired_server_ids": [f"server-{rank}"], "token_count": 2, "token_ids": [11, 12],
                    "global_steps": version, "min_version": version, "max_version": version, "stop_reason": "length",
                } for index in range(concurrency)],
            } for rank in (ranks if selected_ranks is None else selected_ranks)],
        }

    counts = {str(rank): worlds[ranks.index(rank)] for rank in active}
    return {
        "schema_version": 1, "scenario": scenario, "state": "PASSED",
        "training": {"state": "COMPLETED", "completed": True, "completed_steps": 2,
                     "target_steps": 2, "current_param_version": 2},
        "topology": topology, "bootstrap_versions": {str(rank): 0 for rank in ranks},
        "normal_syncs": [{
            "version": 2, "origin": "optimizer_loop", "replica_ranks": list(active),
            "synchronized_versions": {str(rank): 2 for rank in active}, "replica_worker_counts": counts,
            "parameter_validation": {
                "state": "PARAMETERS_VALIDATED", "source_state": "SOURCE_TO_RECEIVER_VALIDATED", "version": 2,
                "worker_count": sum(counts.values()), "parameter_count": 9, "total_numel": 36,
                "manifest_digest": "sha256-parameters", "source_manifest_digest": "sha256-parameters",
            },
        }],
        "generation_before": generation(0), "generation_after": generation(2),
        "training_audits": [{"replica_rank": rank, "completed": 2, "token_count": 4, "inflight": 0, "failed_calls": 0}
                            for rank in active],
        "cleanup": {"test_resources_released": True, "ce_unregistered": True, "lb_removed": True,
                    "donor_pg_preserved": True, "donor_restored": True, "donor_sync_version": 2,
                    "borrowers": [{"replica_rank": rank, "state": "TEST_RUNTIME_CLEANED", "errors": [],
                                   "release_confirmed": True, "actors_dead": True,
                                   "child_processes_dead": True, "endpoint_closed": True} for rank in ranks],
                    "donor_parameter_validation": {
                        "state": "PARAMETERS_VALIDATED", "source_state": "SOURCE_TO_RECEIVER_VALIDATED",
                        "version": 2, "worker_count": 4, "manifest_digest": "donor-sha", "source_manifest_digest": "donor-sha",
                    },
                    "donor_generation": generation(2, topology["donor_ranks"])},
    }


def _log(receipt):
    return "native diagnostic\n(Runner pid=123) " + verdict.RESULT_MARKER + json.dumps(receipt) + "\n"


@pytest.mark.parametrize("scenario", sorted(verdict.SCENARIOS))
def test_complete_receipt_for_each_supported_scenario(scenario):
    receipt = _receipt(scenario)
    assert verdict.validate_log(_log(receipt), scenario, 0) == receipt


@pytest.mark.parametrize("exit_status", [1, -9, 137, None, False, "0"])
def test_success_receipt_cannot_override_failed_or_unknown_process(exit_status):
    with pytest.raises(verdict.E2EVerdictError, match="exit status"):
        verdict.validate_log(_log(_receipt()), "basic", exit_status)


@pytest.mark.parametrize("log", [
    'MULTITASK_TRAINING_COMPLETE {"state":"COMPLETED","completed":true}\n'
    'CE_PARAMETER_VALIDATION {"state":"PARAMETERS_VALIDATED"}\nD4_RUNTIME_RESULT {"state":"LB_READY"}',
    verdict.RESULT_MARKER + "{bad json}",
    verdict.RESULT_MARKER + json.dumps(_receipt()) + " trailing text",
    _log(_receipt()) + _log(_receipt()),
    verdict.RESULT_MARKER + '{"state":"FAILED","state":"PASSED"}',
])
def test_marker_fragments_malformed_json_and_combined_runs_cannot_pass(log):
    with pytest.raises(verdict.E2EVerdictError):
        verdict.validate_log(log, "basic", 0)


@pytest.mark.parametrize(("path", "value"), [
    (("schema_version",), True),
    (("state",), "FAILED"),
    (("scenario",), "split"),
    (("training", "completed"), "true"),
    (("training", "completed_steps"), 1),
    (("training", "target_steps"), 0),
    (("training", "current_param_version"), 0),
    (("topology", "validated"), False),
    (("topology", "world_sizes"), []),
    (("topology", "replica_ranks"), [3, 3]),
    (("topology", "training_replica_ranks"), [2]),
    (("bootstrap_versions",), {"4": 0}),
    (("bootstrap_versions", "3"), 2),
    (("normal_syncs",), []),
    (("normal_syncs", 0, "origin"), "test_explicit_normal_sync"),
    (("normal_syncs", 0, "version"), 3),
    (("normal_syncs", 0, "synchronized_versions", "3"), 1),
    (("normal_syncs", 0, "synchronized_versions"), {"4": 2}),
    (("normal_syncs", 0, "replica_worker_counts", "3"), 3),
    (("normal_syncs", 0, "parameter_validation", "source_state"), None),
    (("normal_syncs", 0, "parameter_validation", "version"), 1),
    (("normal_syncs", 0, "parameter_validation", "worker_count"), 3),
    (("normal_syncs", 0, "parameter_validation", "parameter_count"), 0),
    (("normal_syncs", 0, "parameter_validation", "total_numel"), 0),
    (("normal_syncs", 0, "parameter_validation", "source_manifest_digest"), "other-weights"),
    (("generation_after", "version"), 1),
    (("generation_after", "routes_restored"), False),
    (("generation_after", "inflight_after"), 1),
    (("generation_after", "replicas"), []),
    (("generation_after", "replicas", 0, "server_id"), "different-runtime"),
    (("generation_after", "replicas", 0, "requests", 0, "acquired_server_ids"), ["native-server"]),
    (("generation_after", "replicas", 0, "requests", 0, "token_count"), 0),
    (("generation_after", "replicas", 0, "requests", 0, "token_ids"), [11]),
    (("generation_after", "replicas", 0, "requests", 0, "min_version"), 0),
    (("generation_after", "replicas", 0, "requests", 0, "global_steps"), True),
    (("generation_after", "replicas", 0, "requests", 0, "stop_reason"), "aborted"),
    (("training_audits",), []),
    (("training_audits", 0, "completed"), 0),
    (("training_audits", 0, "token_count"), 0),
    (("training_audits", 0, "failed_calls"), 1),
    (("training_audits", 0, "inflight"), 1),
    (("cleanup", "test_resources_released"), False),
    (("cleanup", "ce_unregistered"), False),
    (("cleanup", "lb_removed"), False),
    (("cleanup", "donor_pg_preserved"), False),
    (("cleanup", "donor_restored"), False),
    (("cleanup", "donor_sync_version"), 0),
    (("cleanup", "borrowers"), []),
    (("cleanup", "borrowers", 0, "child_processes_dead"), False),
    (("cleanup", "donor_parameter_validation", "source_manifest_digest"), "old-donor"),
    (("cleanup", "donor_generation", "replicas", 0, "requests", 0, "global_steps"), 0),
])
def test_incomplete_or_contradictory_evidence_fails(path, value):
    receipt = _receipt()
    parent = receipt
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    with pytest.raises(verdict.E2EVerdictError):
        verdict.validate_log(_log(receipt), "basic", 0)


def test_split_requires_generation_and_sync_for_both_replicas():
    for section in ("generation_before", "generation_after"):
        receipt = _receipt("split")
        receipt[section]["replicas"].pop()
        with pytest.raises(verdict.E2EVerdictError, match="every borrowed rank"):
            verdict.validate_result(receipt, "split")
    receipt = _receipt("split")
    sync = receipt["normal_syncs"][0]
    sync["replica_ranks"].pop()
    del sync["synchronized_versions"]["4"]
    del sync["replica_worker_counts"]["4"]
    sync["parameter_validation"]["worker_count"] = 2
    with pytest.raises(verdict.E2EVerdictError, match="every training replica"):
        verdict.validate_result(receipt, "split")


@pytest.mark.parametrize("scenario", ["split", "cross_pg", "merge_world_size"])
def test_smoke_topology_is_not_full_e2e_topology(scenario):
    receipt = _receipt(scenario)
    receipt["topology"]["world_sizes"] = [1] * len(receipt["topology"]["replica_ranks"])
    with pytest.raises(verdict.E2EVerdictError):
        verdict.validate_result(receipt, scenario)


def test_shared_bundle_requires_active_b_training_and_restored_a_generation():
    receipt = _receipt("shared_bundle")
    explicit = copy.deepcopy(receipt["normal_syncs"][0])
    explicit.update(origin="test_explicit_normal_sync", version=0, replica_ranks=[3],
                    synchronized_versions={"3": 0}, replica_worker_counts={"3": 1})
    explicit["parameter_validation"]["version"] = 0
    receipt["normal_syncs"].insert(0, explicit)
    verdict.validate_result(receipt, "shared_bundle")
    receipt["normal_syncs"].pop()
    with pytest.raises(verdict.E2EVerdictError, match="optimizer-loop"):
        verdict.validate_result(receipt, "shared_bundle")


@pytest.mark.parametrize("scenario", ["idempotent", "concurrent_idempotent"])
@pytest.mark.parametrize("field", ["same_rank", "same_server", "one_runtime"])
def test_idempotency_needs_rank_server_and_runtime_evidence(scenario, field):
    receipt = _receipt(scenario)
    receipt["topology"]["idempotency"][field] = False
    with pytest.raises(verdict.E2EVerdictError, match="one rank/server/runtime"):
        verdict.validate_result(receipt, scenario)


def test_pressure_requires_concurrent_distinct_requests_in_each_phase():
    receipt = _receipt("pressure")
    receipt["generation_after"]["concurrency"] = 1
    with pytest.raises(verdict.E2EVerdictError, match="concurrency"):
        verdict.validate_result(receipt, "pressure")
    receipt = _receipt("pressure")
    receipt["generation_before"]["replicas"][0]["requests"].pop()
    with pytest.raises(verdict.E2EVerdictError, match=">=4 requests"):
        verdict.validate_result(receipt, "pressure")


def test_request_reuse_cannot_masquerade_as_after_generation():
    receipt = _receipt()
    receipt["generation_after"]["replicas"][0]["requests"][0]["request_id"] = "req-0-3-0"
    with pytest.raises(verdict.E2EVerdictError, match="reused"):
        verdict.validate_result(receipt, "basic")


def test_manifest_may_include_other_native_workers_but_must_cover_them_exactly():
    receipt = _receipt()
    sync = receipt["normal_syncs"][0]
    sync["replica_ranks"].append(0)
    sync["synchronized_versions"]["0"] = 2
    sync["replica_worker_counts"]["0"] = 4
    sync["parameter_validation"]["worker_count"] = 8
    verdict.validate_result(receipt, "basic")
    sync["parameter_validation"]["worker_count"] = 4
    with pytest.raises(verdict.E2EVerdictError, match="effective worker set"):
        verdict.validate_result(receipt, "basic")


def test_cli_runs_without_project_imports_and_requires_actual_exit_status(tmp_path):
    log = tmp_path / "native.log"
    log.write_text(_log(_receipt()), encoding="utf-8")
    command = [sys.executable, "-I", str(MODULE_PATH), str(log), "basic"]
    passed = subprocess.run(command + ["--process-exit-code", "0"], capture_output=True, text=True)
    assert passed.returncode == 0, passed.stderr
    assert "E2E PASS: basic" in passed.stdout
    assert verdict.RESULT_MARKER not in passed.stdout
    failed = subprocess.run(command + ["--process-exit-code", "137"], capture_output=True, text=True)
    assert failed.returncode == 1
    assert "exit status" in failed.stderr
    missing = subprocess.run(command, capture_output=True, text=True)
    assert missing.returncode != 0
