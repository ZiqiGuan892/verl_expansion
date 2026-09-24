#!/usr/bin/env bash
#
# D0-D4 串行批量验收入口。
#
# 每次只调用 D0_D4_comprehensive_test.sh 的一个场景，等待该进程完全退出
# 后再启动下一个场景。前一个场景失败、未完成或被阻塞时仍继续执行后续
# 场景，最终在 results.tsv/summary.json 中记录每个场景的结果。
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SINGLE_TEST="${SCRIPT_DIR}/D0_D4_comprehensive_test.sh"
[ -f "${SINGLE_TEST}" ] || {
    echo "未找到单场景脚本：${SINGLE_TEST}" >&2
    exit 2
}
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHON_BIN
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || {
    echo "找不到 Python 解释器：${PYTHON_BIN}" >&2
    exit 2
}

DEFAULT_SCENARIOS="S0 S1 S2 S3 S4 S5 S6 S7 S8 S9 S10 S11 S12 S13 S14 S15 S16"
SCENARIOS_RAW="${D0_D4_BATCH_SCENARIOS:-${DEFAULT_SCENARIOS}}"
SCENARIOS="$(printf '%s' "${SCENARIOS_RAW}" | tr ',' ' ')"
RUN_ID="$(date +%Y%m%d%H%M%S)"
ROOT_LOG_DIR="${D0_D4_BATCH_LOG_DIR:-${VERL_REPO_DIR:-${SCRIPT_DIR}}/logs/d0_d4_batch}/${RUN_ID}"
mkdir -p "${ROOT_LOG_DIR}"
RESULTS_FILE="${ROOT_LOG_DIR}/results.tsv"
SUMMARY_FILE="${ROOT_LOG_DIR}/summary.json"
: > "${RESULTS_FILE}"

safe_detail() {
    printf '%s' "$1" | tr '\t\r\n' '   '
}

record_batch_result() {
    scenario="$1"
    status="$2"
    log_file="$3"
    detail="$(safe_detail "$4")"
    printf '%s\t%s\t%s\t%s\n' "${scenario}" "${status}" "${log_file}" "${detail}" >> "${RESULTS_FILE}"
    echo "[D0-D4-BATCH] ${scenario}: ${status} (${detail})"
    [ -n "${log_file}" ] && echo "[D0-D4-BATCH] log: ${log_file}"
}

for scenario in ${SCENARIOS}; do
    case "${scenario}" in
        S[0-9]|S1[0-6]) ;;
        *)
            record_batch_result "${scenario}" BLOCKED "" "场景名必须为 S0 到 S16"
            continue
            ;;
    esac

    scenario_dir="${ROOT_LOG_DIR}/${scenario}"
    mkdir -p "${scenario_dir}"
    log_file="${scenario_dir}/driver.log"
    echo "[D0-D4-BATCH] 开始 ${scenario}；前一个场景已退出，当前场景独立运行"

    set +e
    env \
        D0_D4_SCENARIOS="${scenario}" \
        D0_D4_REQUIRE_COMPLETE="1" \
        D0_D4_LOG_DIR="${scenario_dir}" \
        bash "${SINGLE_TEST}" 2>&1 | tee "${log_file}"
    command_statuses=( "${PIPESTATUS[@]}" )
    set -e
    child_status="${command_statuses[0]}"
    tee_status="${command_statuses[1]}"

    summary_file="$(grep -F '[D0-D4] summary: ' "${log_file}" | tail -n 1 | sed 's/.*summary: //')"
    scenario_status="FAIL"
    detail="单场景脚本未生成 summary"
    if [ -n "${summary_file}" ] && [ -f "${summary_file}" ]; then
        set +e
        scenario_status="$(${PYTHON_BIN} - "${summary_file}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    summary = json.load(stream)
results = summary.get("results", [])
status = results[0].get("status", "FAIL") if results else "FAIL"
print(status)
PY
        )"
        status_parse=$?
        detail="$(${PYTHON_BIN} - "${summary_file}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    results = json.load(stream).get("results", [])
print(results[0].get("detail", "") if results else "没有结果记录")
PY
        )"
        detail_parse=$?
        set -e
        if [ "${status_parse}" -ne 0 ] || [ "${detail_parse}" -ne 0 ] || [ -z "${scenario_status}" ]; then
            scenario_status="FAIL"
            detail="summary.json 无法解析"
        fi
    elif [ "${child_status}" -ne 0 ]; then
        detail="单场景进程退出码=${child_status}"
    elif [ "${tee_status}" -ne 0 ]; then
        detail="日志写入失败，tee exit=${tee_status}"
    fi

    # 子进程无论成功或失败都已退出；只记录结果，继续下一个场景。
    if [ "${child_status}" -ne 0 ] && [ "${scenario_status}" = "PASS" ]; then
        scenario_status="FAIL"
        detail="单场景进程退出码=${child_status}"
    fi
    if [ "${tee_status}" -ne 0 ]; then
        scenario_status="FAIL"
        detail="日志写入失败，tee exit=${tee_status}"
    fi
    record_batch_result "${scenario}" "${scenario_status}" "${log_file}" "${detail}"
done

export D0_D4_BATCH_RESULTS_FILE="${RESULTS_FILE}"
export D0_D4_BATCH_SUMMARY_FILE="${SUMMARY_FILE}"
"${PYTHON_BIN}" - <<'PY'
import json
import os

rows = []
with open(os.environ["D0_D4_BATCH_RESULTS_FILE"], encoding="utf-8") as stream:
    for line in stream:
        scenario, status, log_file, detail = line.rstrip("\n").split("\t", 3)
        rows.append({"scenario": scenario, "status": status, "log": log_file, "detail": detail})

summary = {
    "results": rows,
    "passed": sum(item["status"] == "PASS" for item in rows),
    "failed": sum(item["status"] == "FAIL" for item in rows),
    "incomplete": sum(item["status"] == "INCOMPLETE" for item in rows),
    "blocked": sum(item["status"] == "BLOCKED" for item in rows),
}
with open(os.environ["D0_D4_BATCH_SUMMARY_FILE"], "w", encoding="utf-8") as stream:
    json.dump(summary, stream, ensure_ascii=False, indent=2, sort_keys=True)
    stream.write("\n")
print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
PY

echo "[D0-D4-BATCH] results: ${RESULTS_FILE}"
echo "[D0-D4-BATCH] summary: ${SUMMARY_FILE}"

if grep -Eq $'^([^\t]+)\tFAIL\t' "${RESULTS_FILE}"; then
    exit 1
fi
if grep -Eq $'^([^\t]+)\t(INCOMPLETE|BLOCKED)\t' "${RESULTS_FILE}"; then
    exit 2
fi
exit 0
