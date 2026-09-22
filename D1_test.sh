#!/usr/bin/env bash
#
# D1 创建契约测试入口，路径和 Ascend 环境参照已在服务器跑通的 multi_task_run.sh。
#
# 两个脚本放在相同目录，服务器布局为：
#   VERL_REPO_DIR/
#   ├── multi_task_run.sh
#   ├── D1_test.sh
#   └── verl/
#       ├── verl/                  # 原生 Python 包
#       └── multi_task_verl/        # 插件仓库，包含 src/ 和 tests/
#
# D1 只验证契约、rank、幂等和预留回执，不启动训练或创建 borrowed runtime。
# 延续旧 Bash 兼容要求，不依赖 pipefail；末尾显式检查 pytest 和 tee 的退出码。
set -eu
set -x

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export VERL_REPO_DIR="${VERL_REPO_DIR:-${SCRIPT_DIR}}"
export VERL_MULTI_TASK_ROOT="${VERL_MULTI_TASK_ROOT:-${VERL_REPO_DIR}/verl/multi_task_verl}"
export PYTHONPATH="${VERL_MULTI_TASK_ROOT}/src:${VERL_REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export VERL_SOURCE_ROOT="${VERL_SOURCE_ROOT:-${VERL_REPO_DIR}/verl}"
# 与 multi_task_run.sh 一致，先进入包含原生 verl/ Python 包的仓根。
cd "${VERL_SOURCE_ROOT}"

export PYTHONPATH="${VERL_MULTI_TASK_ROOT}/src:${VERL_SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_DATASETS_CACHE="${VERL_REPO_DIR}/cache"

# ------------------------------ Ascend 环境 ------------------------------

ASCEND_TOOLKIT_ENV="${ASCEND_TOOLKIT_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
ASCEND_ATB_ENV="${ASCEND_ATB_ENV:-/usr/local/Ascend/nnal/atb/set_env.sh}"

[ -f "${ASCEND_TOOLKIT_ENV}" ] || { echo "未找到 ${ASCEND_TOOLKIT_ENV}" >&2; exit 1; }
[ -f "${ASCEND_ATB_ENV}" ] || { echo "未找到 ${ASCEND_ATB_ENV}" >&2; exit 1; }

# 与已跑通的入口一致，加载 Ascend 环境时临时关闭 errexit 和 nounset。
set +e +u
source "${ASCEND_TOOLKIT_ENV}"
source "${ASCEND_ATB_ENV}"
set -eu

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES:-1}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1500}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_HOST_SOCKET_PORT_RANGE:-60000-60050}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-61000-61050}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_ASCEND_ENABLE_NZ="${VLLM_ASCEND_ENABLE_NZ:-0}"
export VLLM_ALLREDUCE_USE_SYMM_MEM="${VLLM_ALLREDUCE_USE_SYMM_MEM:-0}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

# ------------------------------ D1 必要测试配置 ------------------------------

# 使用与训练入口相同的 python3；如需已激活环境之外的解释器，可覆盖 MT_PYTHON。
MT_PYTHON="${MT_PYTHON:-python3}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD="${PYTEST_DISABLE_PLUGIN_AUTOLOAD:-1}"
export RAY_USAGE_STATS_ENABLED="${RAY_USAGE_STATS_ENABLED:-0}"

# 测试目标相对插件仓根解析；也可设为 tests/unit 或一个绝对路径。
D1_TEST_TARGET="${D1_TEST_TARGET:-tests/unit/test_borrowed_contract.py}"
[ -d "${VERL_MULTI_TASK_ROOT}/src/multi_task_scheduler" ] || {
    echo "插件源码目录不存在：${VERL_MULTI_TASK_ROOT}/src/multi_task_scheduler" >&2
    exit 1
}
# pytest 需要在插件仓根解析 tests/，不依赖用户从哪个目录调用脚本。
cd "${VERL_MULTI_TASK_ROOT}"
[ -e "${D1_TEST_TARGET%%::*}" ] || {
    echo "D1 测试目标不存在：${D1_TEST_TARGET}；请确认插件已更新到 D1。" >&2
    exit 1
}

# 日志目录默认与 multi_task_run.sh 一致，可单独设置 D1_LOG_DIR。
LOG_DIR="${LOG_DIR:-${VERL_REPO_DIR}/logs}"
D1_LOG_DIR="${D1_LOG_DIR:-${LOG_DIR}}"
mkdir -p "${D1_LOG_DIR}"
D1_LOG_FILE="${D1_LOG_DIR}/d1_test_$(date +%Y%m%d%H%M%S).log"

# ------------------------------ D1 测试命令 ------------------------------

CMD=( "${MT_PYTHON}" -m pytest -q -p no:cacheprovider "${D1_TEST_TARGET}" )
# 与训练入口相同，用命令行参数追加覆盖；这里接收 pytest 参数，例如 -x 或 -vv。
if [ "$#" -gt 0 ]; then
    CMD+=( "$@" )
fi

echo "VERL_REPO_DIR=${VERL_REPO_DIR}"
echo "VERL_SOURCE_ROOT=${VERL_SOURCE_ROOT}"
echo "VERL_MULTI_TASK_ROOT=${VERL_MULTI_TASK_ROOT}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "MT_PYTHON=${MT_PYTHON}"
echo "D1_TEST_TARGET=${D1_TEST_TARGET}"
echo "D1_LOG_FILE=${D1_LOG_FILE}"

# 立即保存整个管道的退出码，避免 tee 成功掩盖 pytest 的失败。
set +e
"${CMD[@]}" 2>&1 | tee "${D1_LOG_FILE}"
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

echo "D1 契约测试通过（未验证 NPU runtime）；日志：${D1_LOG_FILE}"
