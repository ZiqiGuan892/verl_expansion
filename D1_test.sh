#!/usr/bin/env bash
#
# D1 contract test entry point.
#
# The script is intended to live next to async_run.sh and multi_task_run.sh:
#   VERL_REPO_DIR/
#   ├── D1_test.sh
#   └── verl/
#       ├── verl/
#       └── verl_multi_task/ (or multi_task_verl/)
#
# It runs D1's metadata-only tests.  D1 must not start Ray actors, create a
# PlacementGroup, start a CE Worker, or launch a vLLM engine.
# It uses only old-Bash-compatible syntax and does not require pipefail.
set -eu

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"

# Prefer explicit paths. When the script is kept inside this plugin checkout,
# discover the sibling native checkout used by the local development tree:
#   workspace/verl-multi-task/D1_test.sh
#   workspace/verl/verl/experimental/fully_async_policy/fully_async_main.py
if [ -z "${VERL_REPO_DIR:-}" ] && [ -z "${VERL_SOURCE_ROOT:-}" ] && [ -z "${VERL_MULTI_TASK_ROOT:-}" ]; then
    if [ -f "${SCRIPT_DIR}/../verl/verl/experimental/fully_async_policy/fully_async_main.py" ]; then
        VERL_REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
        VERL_SOURCE_ROOT="${VERL_REPO_DIR}/verl"
        VERL_MULTI_TASK_ROOT="${SCRIPT_DIR}"
    elif [ -f "${SCRIPT_DIR}/../verl/experimental/fully_async_policy/fully_async_main.py" ]; then
        VERL_REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
        VERL_SOURCE_ROOT="${VERL_REPO_DIR}"
        VERL_MULTI_TASK_ROOT="${SCRIPT_DIR}"
    else
        VERL_REPO_DIR="${SCRIPT_DIR}"
    fi
fi

VERL_REPO_DIR="$(cd -- "${VERL_REPO_DIR:-${SCRIPT_DIR}}" && pwd)"
VERL_SOURCE_ROOT="$(cd -- "${VERL_SOURCE_ROOT:-${VERL_REPO_DIR}/verl}" && pwd)"

# Accept both names used by existing server deployments.
if [ -n "${VERL_MULTI_TASK_ROOT:-}" ]; then
    VERL_MULTI_TASK_ROOT="${VERL_MULTI_TASK_ROOT}"
elif [ -d "${VERL_SOURCE_ROOT}/verl_multi_task" ]; then
    VERL_MULTI_TASK_ROOT="${VERL_SOURCE_ROOT}/verl_multi_task"
elif [ -d "${VERL_SOURCE_ROOT}/multi_task_verl" ]; then
    VERL_MULTI_TASK_ROOT="${VERL_SOURCE_ROOT}/multi_task_verl"
else
    echo "未找到插件目录：${VERL_SOURCE_ROOT}/verl_multi_task 或 ${VERL_SOURCE_ROOT}/multi_task_verl" >&2
    exit 1
fi

# Convert a user-supplied relative plugin path before changing directories.
case "${VERL_MULTI_TASK_ROOT}" in
    /*) ;;
    *) VERL_MULTI_TASK_ROOT="${PWD}/${VERL_MULTI_TASK_ROOT}" ;;
esac

[ -f "${VERL_SOURCE_ROOT}/verl/experimental/fully_async_policy/fully_async_main.py" ] || {
    echo "VERL_SOURCE_ROOT 不是有效的 verl 源码根目录：${VERL_SOURCE_ROOT}" >&2
    exit 1
}
[ -d "${VERL_MULTI_TASK_ROOT}/src/multi_task_scheduler" ] || {
    echo "插件源码目录不存在：${VERL_MULTI_TASK_ROOT}/src/multi_task_scheduler" >&2
    exit 1
}

export VERL_REPO_DIR VERL_SOURCE_ROOT VERL_MULTI_TASK_ROOT
export PYTHONPATH="${VERL_MULTI_TASK_ROOT}/src:${VERL_SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD="${PYTEST_DISABLE_PLUGIN_AUTOLOAD:-1}"
export RAY_USAGE_STATS_ENABLED="${RAY_USAGE_STATS_ENABLED:-0}"

MT_PYTHON="${MT_PYTHON:-python3}"
D1_TEST_TARGET="${D1_TEST_TARGET:-tests/unit/test_borrowed_contract.py}"
D1_LOG_DIR="${D1_LOG_DIR:-${VERL_MULTI_TASK_ROOT}/logs}"
mkdir -p "${D1_LOG_DIR}"
D1_LOG_FILE="${D1_LOG_DIR}/d1_test_$(date +%Y%m%d%H%M%S).log"

echo "VERL_REPO_DIR=${VERL_REPO_DIR}"
echo "VERL_SOURCE_ROOT=${VERL_SOURCE_ROOT}"
echo "VERL_MULTI_TASK_ROOT=${VERL_MULTI_TASK_ROOT}"
echo "MT_PYTHON=${MT_PYTHON}"
echo "D1_TEST_TARGET=${D1_TEST_TARGET}"
echo "D1_LOG_FILE=${D1_LOG_FILE}"

cd "${VERL_MULTI_TASK_ROOT}"

# Do not use pipefail: older server Bash versions do not support it.  Capture
# the pytest status immediately so tee cannot hide a failing test.
set +e
"${MT_PYTHON}" -m pytest -q -p no:cacheprovider "${D1_TEST_TARGET}" 2>&1 | tee "${D1_LOG_FILE}"
COMMAND_STATUSES=( "${PIPESTATUS[@]}" )
set -e

if [ "${COMMAND_STATUSES[0]}" -ne 0 ]; then
    echo "D1 测试失败，pytest exit=${COMMAND_STATUSES[0]}；日志：${D1_LOG_FILE}" >&2
    exit "${COMMAND_STATUSES[0]}"
fi
if [ "${COMMAND_STATUSES[1]}" -ne 0 ]; then
    echo "D1 日志写入失败，tee exit=${COMMAND_STATUSES[1]}；日志：${D1_LOG_FILE}" >&2
    exit "${COMMAND_STATUSES[1]}"
fi

echo "D1 测试通过；日志：${D1_LOG_FILE}"
