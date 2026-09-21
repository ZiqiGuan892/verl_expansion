#!/usr/bin/env bash
#
# Multi-task Fully Async 启动入口。
# 复用 async_run.sh 的模型、数据、Ascend、资源和异步参数，
# 只额外启用唯一的 Multi-task runtime profile。
#
# 兼容较旧 Bash，只使用基础错误退出选项。
set -eu

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"

# 两个脚本均位于 VERL_REPO_DIR/，支持在其 verl/ 下运行 bash ../multi_task_run.sh。
# 先解析成绝对路径，再进入原生仓库，避免调用位置影响相对路径。
#   原生源码根：VERL_REPO_DIR/verl
#   插件源码：  VERL_REPO_DIR/verl/verl_multi_task/src
# 显式导出后，async_run.sh 沿用这些目录并统一设置 PYTHONPATH。
VERL_REPO_DIR="$(cd -- "${VERL_REPO_DIR:-${SCRIPT_DIR}}" && pwd)"
VERL_SOURCE_ROOT="$(cd -- "${VERL_SOURCE_ROOT:-${VERL_REPO_DIR}/verl}" && pwd)"
VERL_MULTI_TASK_ROOT="${VERL_MULTI_TASK_ROOT:-${VERL_SOURCE_ROOT}/verl_multi_task}"
# 相对插件路径按调用者的工作目录解析，必须在 cd 之前完成。
case "${VERL_MULTI_TASK_ROOT}" in
    /*) ;;
    *) VERL_MULTI_TASK_ROOT="${PWD}/${VERL_MULTI_TASK_ROOT}" ;;
esac
export VERL_REPO_DIR VERL_SOURCE_ROOT VERL_MULTI_TASK_ROOT

# 等价于先进入第一个 verl/，再执行 bash ../async_run.sh 并追加 profile。
cd "${VERL_SOURCE_ROOT}"

exec bash "${SCRIPT_DIR}/async_run.sh" \
    multitask.runtime.profile=experimental_fully_async_standalone \
    "$@"
