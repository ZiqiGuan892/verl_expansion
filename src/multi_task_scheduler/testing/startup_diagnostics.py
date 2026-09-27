"""Read-only diagnostics and a narrow retry decision for CE startup collisions."""

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess


def is_ce_startup_collision(log):
    conflict = "EADDRINUSE" in log or "address already in use" in log.lower()
    ce_worker = "CheckpointEngineWorker" in log or "CE_RENDEZVOUS_CONFLICT " in log
    initialization = any(text in log for text in ("init_workers", "init_process_group", "TCPStore"))
    training_started = any(text in log for text in (
        "MULTITASK_TRAINING_COMPLETE", "Starting FullyAsyncTrainer", "_fit_update_actor", ".fit()",
        "D0_D4_E2E_START ",
    ))
    return conflict and ce_worker and initialization and not training_started


def diagnose(log):
    ports = set()
    for line in log.splitlines():
        if "EADDRINUSE" in line or "address already in use" in line.lower():
            ports.update(int(value) for value in re.findall(r"\bport\s*[:=]\s*(\d+)", line))
        if "CE_RENDEZVOUS_CONFLICT " in line:
            try:
                value = json.loads(line.split("CE_RENDEZVOUS_CONFLICT ", 1)[1]).get("MASTER_PORT")
                if str(value).isdecimal():
                    ports.add(int(value))
            except (ValueError, TypeError):
                pass
    listeners = {}
    ss = shutil.which("ss")
    for port in sorted(ports):
        if not 1 <= port <= 65535:
            continue
        if ss is None:
            listeners[str(port)] = {"unavailable": "ss is not installed"}
            continue
        try:
            probe = subprocess.run([ss, "-H", "-ltnp", f"sport = :{port}"],
                                   capture_output=True, text=True, timeout=5, check=False)
            listeners[str(port)] = {"exit_code": probe.returncode, "stdout": probe.stdout, "stderr": probe.stderr}
        except (OSError, subprocess.TimeoutExpired) as error:
            listeners[str(port)] = {"unavailable": str(error)}
    return {"retryable_ce_startup_collision": is_ce_startup_collision(log), "listeners_now": listeners,
            "note": "Post-failure snapshot only; an empty listener does not identify the earlier port owner."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_file", type=Path)
    args = parser.parse_args()
    result = diagnose(args.log_file.read_text(encoding="utf-8", errors="replace"))
    print("S0_STARTUP_DIAGNOSTIC " + json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["retryable_ce_startup_collision"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
