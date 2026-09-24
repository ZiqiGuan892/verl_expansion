#!/usr/bin/env bash
#
# D0-D4 综合验收入口。
#
# 该脚本把当前已经存在的 native、D2 runtime、D3 bootstrap 和 D4
# command-chain 入口串起来，并为尚未实现生产故障注入的场景明确返回
# INCOMPLETE/BLOCKED。它不会把缺少实现的场景伪报为通过。
#
# 服务器上的典型用法：
#   cd "$VERL_REPO_DIR/verl"
#   D0_D4_SCENARIOS=S0 bash ../D0_D4_comprehensive_test.sh
#   D0_D4_SCENARIOS=S1 bash ../D0_D4_comprehensive_test.sh
#
# 旧 Bash 兼容：不依赖 pipefail；每个子进程使用 tee 实时打印并记录
# 日志，通过 PIPESTATUS 显式检查退出码。完整验收返回 0；执行失败返回 1；代码或环境尚未支持
# 选定场景返回 2。设置 D0_D4_REQUIRE_COMPLETE=0 可用于阶段性回归，
# 但输出中的 INCOMPLETE/BLOCKED 仍然表示综合验收未完成。
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 同时兼容服务器布局：脚本在 VERL_REPO_DIR、插件在 verl/multi_task_verl；
# 以及本地开发布局：脚本就在插件仓库根目录。
if [ -d "${SCRIPT_DIR}/src/multi_task_scheduler" ]; then
    DEFAULT_MULTI_TASK_ROOT="${SCRIPT_DIR}"
    DEFAULT_REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
    DEFAULT_SOURCE_ROOT="${DEFAULT_REPO_DIR}/verl"
else
    DEFAULT_REPO_DIR="${SCRIPT_DIR}"
    DEFAULT_SOURCE_ROOT="${SCRIPT_DIR}/verl"
    DEFAULT_MULTI_TASK_ROOT="${SCRIPT_DIR}/verl/multi_task_verl"
fi

export VERL_REPO_DIR="${VERL_REPO_DIR:-${DEFAULT_REPO_DIR}}"
export VERL_SOURCE_ROOT="${VERL_SOURCE_ROOT:-${DEFAULT_SOURCE_ROOT}}"
export VERL_MULTI_TASK_ROOT="${VERL_MULTI_TASK_ROOT:-${DEFAULT_MULTI_TASK_ROOT}}"

[ -d "${VERL_MULTI_TASK_ROOT}" ] || {
    echo "插件仓库不存在：${VERL_MULTI_TASK_ROOT}" >&2
    exit 2
}
[ -d "${VERL_SOURCE_ROOT}" ] || {
    echo "原生 verl 源码根不存在：${VERL_SOURCE_ROOT}" >&2
    exit 2
}
[ -f "${SCRIPT_DIR}/multi_task_run.sh" ] || {
    echo "未找到 multi_task_run.sh：${SCRIPT_DIR}/multi_task_run.sh" >&2
    exit 2
}

PYTHON_BIN="${PYTHON_BIN:-python3}"
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || {
    echo "找不到 Python 解释器：${PYTHON_BIN}" >&2
    exit 2
}

# 一次只执行一个场景。需要执行多个场景时，由外层 shell 逐次调用本脚本，
# 这样每个场景拥有独立 Ray 会话、日志目录和显存清理边界。
export D0_D4_SCENARIOS="${D0_D4_SCENARIOS:-S0}"
export D0_D4_REQUIRE_COMPLETE="${D0_D4_REQUIRE_COMPLETE:-1}"
RUN_ID="$(date +%Y%m%d%H%M%S)"
RUN_DIR="${D0_D4_LOG_DIR:-${VERL_REPO_DIR}/logs/d0_d4_comprehensive}/${RUN_ID}"
mkdir -p "${RUN_DIR}"
RESULTS_FILE="${RUN_DIR}/results.tsv"
SUMMARY_FILE="${RUN_DIR}/summary.json"
: > "${RESULTS_FILE}"

safe_detail() {
    # 结果文件用 TSV 保存，避免 detail 中出现换行或制表符。
    printf '%s' "$1" | tr '\t\r\n' '   '
}

record_result() {
    scenario="$1"
    status="$2"
    log_file="$3"
    detail="$(safe_detail "$4")"
    printf '%s\t%s\t%s\t%s\n' "${scenario}" "${status}" "${log_file}" "${detail}" >> "${RESULTS_FILE}"
    echo "[D0-D4] ${scenario}: ${status} (${detail})"
    if [ -n "${log_file}" ]; then
        echo "[D0-D4] log: ${log_file}"
    fi
}

run_logged() {
    log_file="$1"
    shift
    set +e
    # 不能把 stdout/stderr 重定向到文件，否则 native verl 的启动、Ray、
    # vLLM 和训练日志只会在子进程结束后才能看到。tee 同时保留实时终端输出。
    "$@" 2>&1 | tee "${log_file}"
    command_statuses=( "${PIPESTATUS[@]}" )
    set -e
    if [ "${command_statuses[0]}" -ne 0 ]; then
        return "${command_statuses[0]}"
    fi
    return "${command_statuses[1]}"
}

has_error_marker() {
    log_file="$1"
    grep -Eq 'Traceback|RayTaskError|AssertionError|Engine core initialization failed|OutOfMemory|OOM|HCCL.*parameter error' "${log_file}"
}

write_environment() {
    plugin_commit="$(git -C "${VERL_MULTI_TASK_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
    verl_commit="$(git -C "${VERL_SOURCE_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
    export D0_D4_ENV_FILE="${RUN_DIR}/environment.json"
    export D0_D4_PLUGIN_COMMIT="${plugin_commit}"
    export D0_D4_VERL_COMMIT="${verl_commit}"
    export D0_D4_RUN_ID="${RUN_ID}"
    "${PYTHON_BIN}" - <<'PY' > "${D0_D4_ENV_FILE}"
import importlib.metadata
import json
import os
import platform
import sys

def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None

visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
print(json.dumps({
    "run_id": os.environ.get("D0_D4_RUN_ID"),
    "plugin_commit": os.environ.get("D0_D4_PLUGIN_COMMIT"),
    "verl_commit": os.environ.get("D0_D4_VERL_COMMIT"),
    "python": sys.executable,
    "python_version": platform.python_version(),
    "ray": package_version("ray"),
    "vllm": package_version("vllm"),
    "vllm_ascend": package_version("vllm-ascend"),
    "device_type": os.environ.get("DEVICE_TYPE", "npu"),
    "visible_devices": [item for item in visible.split(",") if item],
    "verl_repo_dir": os.environ.get("VERL_REPO_DIR"),
    "verl_source_root": os.environ.get("VERL_SOURCE_ROOT"),
    "multi_task_root": os.environ.get("VERL_MULTI_TASK_ROOT"),
    "model_path": os.environ.get("MODEL_PATH", os.environ.get("ACTOR_MODEL_PATH")),
    "train_file": os.environ.get("TRAIN_FILE"),
    "test_file": os.environ.get("TEST_FILE"),
}, indent=2, sort_keys=True))
PY
    echo "[D0-D4] environment: ${D0_D4_ENV_FILE}"
}

run_native_baseline() {
    log_file="${RUN_DIR}/S0_native_baseline.log"
    echo "[D0-D4] 开始 S0 native baseline"
    if run_logged "${log_file}" bash "${SCRIPT_DIR}/multi_task_run.sh" \
        "actor_rollout_ref.actor.ppo_mini_batch_size=2" \
        "actor_rollout_ref.rollout.n=2" \
        "async_training.require_batches=1" \
        "async_training.trigger_parameter_sync_step=1" \
        "trainer.total_training_steps=1" \
        "trainer.total_epochs=1" \
        "rollout.total_rollout_steps=2" \
        "+actor_rollout_ref.rollout.enable_sleep_mode=true" \
        "actor_rollout_ref.rollout.free_cache_engine=true"; then
        if ! has_error_marker "${log_file}" && \
            (grep -Fq '[ASYNC MAIN] Training completed or interrupted' "${log_file}" || \
                grep -Fq 'total time:' "${log_file}" || \
                grep -Fq '[ASYNC MAIN] One component completed successfully' "${log_file}"); then
            record_result S0 PASS "${log_file}" "native main_ppo 完成且未发现异常标记"
        else
            if has_error_marker "${log_file}"; then
                record_result S0 FAIL "${log_file}" "native 日志包含异常标记，详见实时输出和日志文件"
            else
                record_result S0 FAIL "${log_file}" "native 进程退出为 0，但缺少完成标记；请检查实时日志末尾"
            fi
        fi
    else
        record_result S0 FAIL "${log_file}" "native main_ppo 退出失败"
    fi
}

run_d4_scenario() {
    scenario="$1"
    d4_scenario="$2"
    coverage="$3"
    log_file="${RUN_DIR}/${scenario}_d4.log"
    child_log_dir="${RUN_DIR}/${scenario}_d4_logs"
    mkdir -p "${child_log_dir}"
    echo "[D0-D4] 开始 ${scenario} -> D4 ${d4_scenario}"
    if run_logged "${log_file}" env \
        VERL_REPO_DIR="${VERL_REPO_DIR}" \
        VERL_SOURCE_ROOT="${VERL_SOURCE_ROOT}" \
        VERL_MULTI_TASK_ROOT="${VERL_MULTI_TASK_ROOT}" \
        D4_RUNTIME_SCENARIOS="${d4_scenario}" \
        D4_RUNTIME_LOG_DIR="${child_log_dir}" \
        bash "${SCRIPT_DIR}/D4_test.sh"; then
        if grep -Fq 'D4_RUNTIME_RESULT' "${log_file}" && \
            grep -Fq '"state": "LB_READY"' "${log_file}" && \
            grep -Fq 'D4_RUNTIME_CLEANUP' "${log_file}" && \
            ! has_error_marker "${log_file}"; then
            if [ "${coverage}" = "complete" ]; then
                record_result "${scenario}" PASS "${log_file}" "D4 创建、CE bootstrap、LB_READY 和测试清理通过"
            else
                record_result "${scenario}" INCOMPLETE "${log_file}" "当前 D4 只验证 endpoint/LB marker，缺少综合设计要求的真实 generate/后续同步或完整拓扑"
            fi
        else
            record_result "${scenario}" FAIL "${log_file}" "D4 日志缺少 LB_READY、cleanup 或 runtime receipt"
        fi
    else
        record_result "${scenario}" FAIL "${log_file}" "D4 main_ppo 进程失败"
    fi
}

run_d2_negative() {
    scenario="$1"
    d2_scenarios="$2"
    log_file="${RUN_DIR}/${scenario}_d2_negative.log"
    child_log_dir="${RUN_DIR}/${scenario}_d2_logs"
    mkdir -p "${child_log_dir}"
    echo "[D0-D4] 开始 ${scenario} -> D2 negative ${d2_scenarios}"
    if run_logged "${log_file}" env \
        VERL_REPO_DIR="${VERL_REPO_DIR}" \
        VERL_SOURCE_ROOT="${VERL_SOURCE_ROOT}" \
        VERL_MULTI_TASK_ROOT="${VERL_MULTI_TASK_ROOT}" \
        D2_RUNTIME_SCENARIOS="${d2_scenarios}" \
        D2_RUNTIME_LOG_DIR="${child_log_dir}" \
        bash "${SCRIPT_DIR}/D2_runtime_test.sh"; then
        if grep -Fq 'D2_RUNTIME_RESULT' "${log_file}" && \
            grep -Fq '"status": "EXPECTED_FAILURE"' "${log_file}" && \
            ! grep -Eq 'Traceback|AssertionError|RayTaskError' "${log_file}"; then
            record_result "${scenario}" PASS "${log_file}" "预期失败场景被拒绝，未发布 RUNTIME_READY"
        else
            record_result "${scenario}" FAIL "${log_file}" "D2 negative 日志缺少预期失败 receipt"
        fi
    else
        record_result "${scenario}" FAIL "${log_file}" "D2 negative main_ppo 进程失败"
    fi
}

run_unit_only() {
    scenario="$1"
    log_file="${RUN_DIR}/unit_contracts.log"
    if [ ! -f "${log_file}" ]; then
        echo "[D0-D4] 为 ${scenario} 执行 D2/D1 单元契约测试"
        if ! run_logged "${log_file}" env \
            VERL_REPO_DIR="${VERL_REPO_DIR}" \
            VERL_SOURCE_ROOT="${VERL_SOURCE_ROOT}" \
            VERL_MULTI_TASK_ROOT="${VERL_MULTI_TASK_ROOT}" \
            D2_LOG_DIR="${RUN_DIR}/unit_logs" \
            bash "${SCRIPT_DIR}/D2_test.sh"; then
            record_result "${scenario}" FAIL "${log_file}" "契约单元测试失败"
            return
        fi
    fi
    record_result "${scenario}" INCOMPLETE "${log_file}" "仅有单元契约证据，真实 Ray 并发/重试场景尚未实现"
}

mark_blocked() {
    scenario="$1"
    record_result "${scenario}" BLOCKED "" "当前代码没有对应的真实故障注入或请求压力测试入口"
}

write_environment

case "${D0_D4_SCENARIOS}" in
    ""|*,*|*[!a-zA-Z0-9_]*)
        echo "D0_D4_SCENARIOS 必须是一个场景名，例如 S0；一次不能传入多个场景。" >&2
        exit 2
        ;;
esac

scenario="${D0_D4_SCENARIOS}"
case "${scenario}" in
    S0) run_native_baseline ;;
    S1) run_d4_scenario S1 basic partial ;;
    S2) run_d4_scenario S2 split partial ;;
    S3) run_d4_scenario S3 cross_pg partial ;;
    S4) run_d4_scenario S4 fragmented partial ;;
    S8|S9) run_unit_only "${scenario}" ;;
    S10) run_d2_negative S10 expired ;;
    S11) run_d2_negative S11 missing_pg,duplicate_device ;;
    S12|S13|S14|S16) mark_blocked "${scenario}" ;;
    *) record_result "${scenario}" BLOCKED "" "未知综合验收场景" ;;
esac

export D0_D4_RESULTS_FILE="${RESULTS_FILE}"
export D0_D4_SUMMARY_FILE="${SUMMARY_FILE}"
export D0_D4_REQUIRE_COMPLETE
"${PYTHON_BIN}" - <<'PY'
import json
import os

rows = []
with open(os.environ["D0_D4_RESULTS_FILE"], encoding="utf-8") as stream:
    for line in stream:
        scenario, status, log_file, detail = line.rstrip("\n").split("\t", 3)
        rows.append({"scenario": scenario, "status": status, "log": log_file, "detail": detail})

summary = {
    "run_id": os.path.basename(os.path.dirname(os.environ["D0_D4_RESULTS_FILE"])),
    "require_complete": os.environ.get("D0_D4_REQUIRE_COMPLETE", "1") == "1",
    "results": rows,
    "passed": sum(item["status"] == "PASS" for item in rows),
    "failed": sum(item["status"] == "FAIL" for item in rows),
    "incomplete": sum(item["status"] == "INCOMPLETE" for item in rows),
    "blocked": sum(item["status"] == "BLOCKED" for item in rows),
}
with open(os.environ["D0_D4_SUMMARY_FILE"], "w", encoding="utf-8") as stream:
    json.dump(summary, stream, ensure_ascii=False, indent=2, sort_keys=True)
    stream.write("\n")
print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
PY

echo "[D0-D4] environment: ${RUN_DIR}/environment.json"
echo "[D0-D4] summary: ${SUMMARY_FILE}"

if grep -Eq $'^([^\t]+)\tFAIL\t' "${RESULTS_FILE}"; then
    echo "[D0-D4] 综合验收失败：至少一个场景执行失败。" >&2
    exit 1
fi
if [ "${D0_D4_REQUIRE_COMPLETE}" = "1" ] && grep -Eq $'^([^\t]+)\t(INCOMPLETE|BLOCKED)\t' "${RESULTS_FILE}"; then
    echo "[D0-D4] 综合验收未完成：存在 INCOMPLETE/BLOCKED 场景。" >&2
    exit 2
fi

if grep -Eq $'^([^\t]+)\t(INCOMPLETE|BLOCKED)\t' "${RESULTS_FILE}"; then
    echo "[D0-D4] 阶段回归完成，但综合验收仍有 INCOMPLETE/BLOCKED 场景。"
    exit 0
fi

echo "[D0-D4] 选定场景全部达到当前验收门槛。"
exit 0
