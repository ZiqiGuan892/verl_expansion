#!/usr/bin/env bash
#
# D2 borrowed replica 测试入口。
#
# 服务器目录布局：
#   VERL_REPO_DIR/
#   ├── multi_task_run.sh
#   ├── D2_test.sh
#   └── verl/
#       ├── verl/                  # 原生 verl Python 源码根
#       └── multi_task_verl/        # 本插件仓库，包含 src/ 和 tests/
#
# 本脚本执行 D2 能自动验证的静态、契约、布局和原生继承测试。
# 它不会伪造 Ray/GPU 结果，也不会自动创建 donor PG；真实 borrowed
# runtime 验收需要按照 docs/D2_develop.md 在服务器上由 TaskRunner 提供
# placement claims 后调用 create_borrowed_replica(spec)。
#
# 兼容服务器上的旧 Bash：不使用 pipefail；通过 PIPESTATUS 显式传播
# pytest 和 tee 的退出码。
set -eu
set -x

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export VERL_REPO_DIR="${VERL_REPO_DIR:-${SCRIPT_DIR}}"
export VERL_SOURCE_ROOT="${VERL_SOURCE_ROOT:-${VERL_REPO_DIR}/verl}"
export VERL_MULTI_TASK_ROOT="${VERL_MULTI_TASK_ROOT:-${VERL_SOURCE_ROOT}/multi_task_verl}"

# 不把 VERL_REPO_DIR 加入 PYTHONPATH，避免仓根下可能存在的 vllm/、torch/
# 等目录遮蔽已安装的后端包。Ray 子进程也会继承这两个源码根。
export PYTHONPATH="${VERL_MULTI_TASK_ROOT}/src:${VERL_SOURCE_ROOT}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${VERL_REPO_DIR}/cache}"

# 与 multi_task_run.sh 保持相同的工作目录：原生 verl 源码根。
[ -d "${VERL_SOURCE_ROOT}" ] || {
    echo "原生 verl 源码目录不存在：${VERL_SOURCE_ROOT}" >&2
    exit 1
}
[ -d "${VERL_MULTI_TASK_ROOT}/src/multi_task_scheduler" ] || {
    echo "插件源码目录不存在：${VERL_MULTI_TASK_ROOT}/src/multi_task_scheduler" >&2
    exit 1
}
cd "${VERL_SOURCE_ROOT}"

# ------------------------------ Ascend 环境 ------------------------------

# D2 的真实服务器通常使用 Ascend 环境；设置 D2_LOAD_ASCEND_ENV=0 可在
# 已经由外层作业加载环境时跳过 source。默认值与 multi_task_run.sh 一致。
ASCEND_TOOLKIT_ENV="${ASCEND_TOOLKIT_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
ASCEND_ATB_ENV="${ASCEND_ATB_ENV:-/usr/local/Ascend/nnal/atb/set_env.sh}"
D2_LOAD_ASCEND_ENV="${D2_LOAD_ASCEND_ENV:-1}"

if [ "${D2_LOAD_ASCEND_ENV}" = "1" ]; then
    [ -f "${ASCEND_TOOLKIT_ENV}" ] || {
        echo "未找到 ${ASCEND_TOOLKIT_ENV}；可设置 D2_LOAD_ASCEND_ENV=0 跳过。" >&2
        exit 1
    }
    [ -f "${ASCEND_ATB_ENV}" ] || {
        echo "未找到 ${ASCEND_ATB_ENV}；可设置 D2_LOAD_ASCEND_ENV=0 跳过。" >&2
        exit 1
    }

    # Ascend 的 set_env.sh 可能引用未定义变量，因此临时关闭 errexit 和
    # nounset；加载完成后恢复本脚本的严格检查。
    set +e +u
    source "${ASCEND_TOOLKIT_ENV}"
    source "${ASCEND_ATB_ENV}"
    set -eu
fi

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

# 使用同一个解释器完成 vLLM 预检和 pytest，避免 pytest 使用的 Python 与
# multi_task_run.sh 启动训练时的 Python 不一致。MT_PYTHON 可覆盖解释器。
PYTHON_BIN="${PYTHON_BIN:-${MT_PYTHON:-python3}}"
MT_PYTHON="${MT_PYTHON:-${PYTHON_BIN}}"

# 原生适配测试会导入 verl/vLLM；先报告实际导入位置，避免把 Python 路径
# 或版本问题误判为 D2 代码问题。设置 D2_INCLUDE_NATIVE=0 时跳过该检查。
D2_INCLUDE_NATIVE="${D2_INCLUDE_NATIVE:-1}"
if [ "${D2_INCLUDE_NATIVE}" = "1" ]; then
    "${MT_PYTHON}" - <<'PY'
import importlib.util
import sys

spec = importlib.util.find_spec("vllm")
print("vllm_preflight_python:", sys.executable)
print("vllm_preflight_spec_origin:", None if spec is None else spec.origin)
print(
    "vllm_preflight_search_locations:",
    None if spec is None or spec.submodule_search_locations is None else list(spec.submodule_search_locations),
)
if spec is None:
    raise SystemExit("未找到 vllm；请在当前 PYTHON_BIN 环境安装匹配的 vllm/vllm-ascend")

import vllm

print("vllm_preflight_file:", getattr(vllm, "__file__", None))
print("vllm_preflight_version:", getattr(vllm, "__version__", None))
if not hasattr(vllm, "LLM"):
    raise SystemExit(
        "当前解释器加载的 vllm 没有 LLM；请检查 PYTHONPATH 是否遮蔽 site-packages，"
        "以及 vllm/vllm-ascend 是否安装在同一 Python 环境"
    )
print("vllm_preflight_LLM: available")
PY
fi

# ------------------------------ 测试配置 ------------------------------

export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD="${PYTEST_DISABLE_PLUGIN_AUTOLOAD:-1}"
export RAY_USAGE_STATS_ENABLED="${RAY_USAGE_STATS_ENABLED:-0}"

# D2 默认覆盖：D1 契约、D2 runtime 规划、manager wiring 和原生适配。
D2_TEST_TARGET="${D2_TEST_TARGET:-tests/unit/test_borrowed_contract.py tests/unit/test_borrowed_runtime.py tests/unit/test_wiring.py}"
D2_NATIVE_TEST_TARGET="${D2_NATIVE_TEST_TARGET:-tests/native_unit/test_native_adapters.py}"

cd "${VERL_MULTI_TASK_ROOT}"

for target in ${D2_TEST_TARGET}; do
    [ -e "${target}" ] || {
        echo "D2 测试目标不存在：${target}" >&2
        exit 1
    }
done
if [ "${D2_INCLUDE_NATIVE}" = "1" ]; then
    [ -e "${D2_NATIVE_TEST_TARGET}" ] || {
        echo "原生适配测试目标不存在：${D2_NATIVE_TEST_TARGET}" >&2
        exit 1
    }
fi

LOG_DIR="${LOG_DIR:-${VERL_REPO_DIR}/logs}"
D2_LOG_DIR="${D2_LOG_DIR:-${LOG_DIR}}"
mkdir -p "${D2_LOG_DIR}"
D2_LOG_FILE="${D2_LOG_DIR}/d2_test_$(date +%Y%m%d%H%M%S).log"

CMD=( "${MT_PYTHON}" -m pytest -q -p no:cacheprovider )
for target in ${D2_TEST_TARGET}; do
    CMD+=( "${target}" )
done
if [ "${D2_INCLUDE_NATIVE}" = "1" ]; then
    CMD+=( "${D2_NATIVE_TEST_TARGET}" )
fi
# 允许服务器通过参数追加 pytest 选项，例如：bash ../D2_test.sh -vv -x。
if [ "$#" -gt 0 ]; then
    CMD+=( "$@" )
fi

echo "VERL_REPO_DIR=${VERL_REPO_DIR}"
echo "VERL_SOURCE_ROOT=${VERL_SOURCE_ROOT}"
echo "VERL_MULTI_TASK_ROOT=${VERL_MULTI_TASK_ROOT}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "MT_PYTHON=${MT_PYTHON}"
echo "D2_TEST_TARGET=${D2_TEST_TARGET}"
echo "D2_INCLUDE_NATIVE=${D2_INCLUDE_NATIVE}"
echo "D2_NATIVE_TEST_TARGET=${D2_NATIVE_TEST_TARGET}"
echo "D2_LOG_FILE=${D2_LOG_FILE}"

# 不使用 pipefail；显式保存两个命令的状态，防止 tee 成功掩盖 pytest 失败。
set +e
"${CMD[@]}" 2>&1 | tee "${D2_LOG_FILE}"
COMMAND_STATUSES=( "${PIPESTATUS[@]}" )
set -e

if [ "${COMMAND_STATUSES[0]}" -ne 0 ]; then
    echo "D2 测试失败，pytest exit=${COMMAND_STATUSES[0]}；日志：${D2_LOG_FILE}" >&2
    exit "${COMMAND_STATUSES[0]}"
fi
if [ "${COMMAND_STATUSES[1]}" -ne 0 ]; then
    echo "D2 日志写入失败，tee exit=${COMMAND_STATUSES[1]}；日志：${D2_LOG_FILE}" >&2
    exit "${COMMAND_STATUSES[1]}"
fi

echo "D2 自动化测试通过；日志：${D2_LOG_FILE}"
echo "注意：该脚本未启动真实 borrowed replica。Ray/GPU/Engine 验收请按 docs/D2_develop.md 执行。"
