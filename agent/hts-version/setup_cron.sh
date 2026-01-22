#!/bin/bash
# 设置crontab定时任务的脚本

set -e

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="$SCRIPT_DIR/run_monitor.sh"

echo "========================================="
echo "HTS版本监控 - Crontab设置工具"
echo "========================================="
echo ""

# 检查run_monitor.sh是否存在
if [ ! -f "$RUN_SCRIPT" ]; then
    echo "错误: 未找到运行脚本 $RUN_SCRIPT"
    exit 1
fi

# 确保脚本可执行
chmod +x "$RUN_SCRIPT"

echo "运行脚本路径: $RUN_SCRIPT"
echo ""

# 提示用户选择执行频率
echo "请选择监控频率:"
echo "1. 每小时执行一次（推荐）"
echo "2. 每30分钟执行一次"
echo "3. 每天执行一次（北京时间早上9点）"
echo "4. 每6小时执行一次"
echo "5. 自定义cron表达式"
echo ""
read -p "请输入选项 (1-5): " choice

case $choice in
    1)
        CRON_SCHEDULE="0 * * * *"
        CRON_DESC="每小时第0分钟"
        ;;
    2)
        CRON_SCHEDULE="*/30 * * * *"
        CRON_DESC="每30分钟"
        ;;
    3)
        CRON_SCHEDULE="0 9 * * *"
        CRON_DESC="每天9:00"
        ;;
    4)
        CRON_SCHEDULE="0 */6 * * *"
        CRON_DESC="每6小时"
        ;;
    5)
        read -p "请输入cron表达式（如: 0 * * * *）: " CRON_SCHEDULE
        CRON_DESC="自定义"
        ;;
    *)
        echo "无效选项"
        exit 1
        ;;
esac

# 可选：设置数据库连接字符串
echo ""
read -p "是否设置数据库连接字符串？(y/n，默认n): " set_dsn

DATABASE_DSN_LINE=""
if [ "$set_dsn" = "y" ] || [ "$set_dsn" = "Y" ]; then
    read -p "请输入DATABASE_DSN（格式: postgresql://user:pass@host:port/database）: " dsn_value
    if [ -n "$dsn_value" ]; then
        DATABASE_DSN_LINE="DATABASE_DSN='$dsn_value' "
    fi
fi

# 生成crontab行
CRON_LINE="$CRON_SCHEDULE ${DATABASE_DSN_LINE}$RUN_SCRIPT"

echo ""
echo "========================================="
echo "即将添加的crontab任务:"
echo "频率: $CRON_DESC"
echo "命令: $CRON_LINE"
echo "========================================="
echo ""

# 询问是否继续
read -p "确认添加此定时任务？(y/n): " confirm

if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "已取消"
    exit 0
fi

# 备份当前的crontab（如果存在）
BACKUP_FILE="/tmp/crontab_backup_$(date +%Y%m%d_%H%M%S).txt"
if crontab -l 2>/dev/null; then
    crontab -l > "$BACKUP_FILE"
    echo "当前crontab已备份到: $BACKUP_FILE"
fi

# 检查是否已存在相同的任务
if crontab -l 2>/dev/null | grep -F "$RUN_SCRIPT" > /dev/null; then
    echo ""
    echo "警告: 检测到已存在监控任务:"
    crontab -l | grep -F "$RUN_SCRIPT"
    echo ""
    read -p "是否替换现有任务？(y/n): " replace

    if [ "$replace" = "y" ] || [ "$replace" = "Y" ]; then
        # 删除旧任务
        (crontab -l 2>/dev/null | grep -v -F "$RUN_SCRIPT") | crontab -
        echo "已删除旧任务"
    else
        echo "保留现有任务，退出"
        exit 0
    fi
fi

# 添加新的crontab任务
(crontab -l 2>/dev/null; echo "$CRON_LINE") | crontab -

echo ""
echo "✓ 定时任务添加成功！"
echo ""
echo "查看当前用户的所有定时任务:"
echo "  crontab -l"
echo ""
echo "查看监控日志:"
if [ -w "/var/log" ]; then
    echo "  tail -f /var/log/hts_monitor.log"
else
    echo "  tail -f ~/logs/hts_monitor.log"
fi
echo ""
echo "手动运行监控脚本:"
echo "  $RUN_SCRIPT"
echo ""
echo "删除此定时任务:"
echo "  crontab -l | grep -v '$RUN_SCRIPT' | crontab -"
echo ""
echo "========================================="