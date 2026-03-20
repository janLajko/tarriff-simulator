#!/bin/bash
# HTS版本监控运行脚本
# 自动处理虚拟环境、依赖安装和脚本执行

set -e  # 遇到错误立即退出

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 项目根目录（monitor.py的上两级目录）
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 日志文件路径
LOG_DIR="/var/log"
LOG_FILE="$LOG_DIR/hts_monitor.log"

# 如果/var/log不可写，使用用户home目录
if [ ! -w "$LOG_DIR" ]; then
    LOG_DIR="$HOME/logs"
    mkdir -p "$LOG_DIR"
    LOG_FILE="$LOG_DIR/hts_monitor.log"
fi

# 定义日志函数，同时输出到终端和文件
log_message() {
    local message="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$message" | tee -a "$LOG_FILE"
}

# 记录开始时间
log_message "========== 开始执行HTS版本监控 =========="
log_message "脚本目录: $SCRIPT_DIR"
log_message "项目根目录: $PROJECT_ROOT"
log_message "日志文件: $LOG_FILE"

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 检查Python3是否存在
if ! command -v python3 &> /dev/null; then
    log_message "错误: 未找到python3"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1)
log_message "Python版本: $PYTHON_VERSION"

# 虚拟环境路径
VENV_PATH="$PROJECT_ROOT/venv"

# 创建或激活虚拟环境
if [ ! -d "$VENV_PATH" ]; then
    log_message "创建虚拟环境: $VENV_PATH"
    python3 -m venv "$VENV_PATH" 2>&1 | tee -a "$LOG_FILE"
fi

# 激活虚拟环境
source "$VENV_PATH/bin/activate"
log_message "虚拟环境已激活"

# 升级pip（静默模式，但显示进度）
log_message "检查pip更新..."
pip install --upgrade pip --quiet

# 安装或更新必要的依赖包
log_message "检查并安装依赖..."

# 必需的包列表
REQUIRED_PACKAGES=(
    "psycopg2-binary"
    "requests"
)

# 检查并安装缺失的包
for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! pip show "$package" &> /dev/null; then
        log_message "安装包: $package"
        pip install "$package" --quiet
    else
        log_message "✓ $package 已安装"
    fi
done

# 如果存在requirements.txt，也安装它
if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
    log_message "从requirements.txt安装依赖"
    pip install -r "$PROJECT_ROOT/requirements.txt" --quiet
fi

# 设置环境变量（如果需要）
# 优先使用环境变量中的DATABASE_DSN，否则使用默认值
if [ -z "$DATABASE_DSN" ]; then
    # 可以在这里设置默认的数据库连接字符串
    # export DATABASE_DSN="postgresql://postgres:Xylx1.t123@34.129.224.77:5432/tariff-simulate"
    log_message "警告: DATABASE_DSN未设置，将使用脚本内默认值"
fi

# 运行监控脚本
log_message "执行监控脚本..."
echo "----------------------------------------"

# 使用python而不是python3（因为已在虚拟环境中）
# 同时输出到终端和日志文件
python "$SCRIPT_DIR/monitor.py" 2>&1 | tee -a "$LOG_FILE"
MONITOR_EXIT_CODE=${PIPESTATUS[0]}

echo "----------------------------------------"

# 记录执行结果
if [ $MONITOR_EXIT_CODE -eq 0 ]; then
    log_message "监控脚本执行成功"
else
    log_message "监控脚本执行失败，退出码: $MONITOR_EXIT_CODE"
fi

# 退出虚拟环境
deactivate 2>/dev/null || true

log_message "========== HTS版本监控执行完成 =========="
echo ""

exit $MONITOR_EXIT_CODE