# HTS版本监控工具

监控美国HTS (Harmonized Tariff Schedule) 版本更新，发现新版本时自动发送Lark通知。

## 功能特性

- 🔄 定时检查HTS版本更新（通过USITC API）
- 📊 PostgreSQL数据库持久化存储版本历史
- 🔔 Lark机器人实时通知新版本
- 📝 详细的运行日志记录
- 🚀 自动化虚拟环境和依赖管理

## 文件说明

```
agent/hts-version/
├── monitor.py        # 主监控脚本
├── run_monitor.sh    # 运行包装脚本（处理虚拟环境）
├── setup_cron.sh     # 交互式crontab设置工具
└── README.md         # 本文档
```

## 快速开始

### 1. 一键设置定时任务

```bash
# 运行交互式设置工具
./setup_cron.sh

# 按提示选择：
# - 执行频率（推荐每小时）
# - 是否配置数据库连接
```

### 2. 手动运行测试

```bash
# 直接运行（使用自动化脚本）
./run_monitor.sh

# 或使用Python直接运行（需要手动激活虚拟环境）
source ../../venv/bin/activate
python monitor.py --dsn postgresql://user:pass@host:port/database
```

### 3. 查看运行日志

```bash
# 如果有/var/log权限
tail -f /var/log/hts_monitor.log

# 否则在用户目录
tail -f ~/logs/hts_monitor.log
```

## 配置说明

### 数据库连接

支持三种配置方式（优先级从高到低）：

1. **环境变量**
```bash
export DATABASE_DSN="postgresql://user:pass@host:port/database"
./run_monitor.sh
```

2. **命令行参数**
```bash
python monitor.py --dsn postgresql://user:pass@host:port/database
```

3. **默认值**
```
postgresql://postgres:Xylx1.t123@34.129.224.77:5432/tariff-simulate
```

### Lark Webhook

当前使用的Webhook URL（硬编码在monitor.py中）：
```
https://open.larksuite.com/open-apis/bot/v2/hook/461e14b8-75ae-49ce-b3c9-73b39d06d659
```

## 数据库表结构

```sql
CREATE TABLE hts_version (
    id BIGSERIAL PRIMARY KEY,
    hts_version_name VARCHAR(50) NOT NULL UNIQUE,  -- 版本名称，如"2026HTSRev1"
    description VARCHAR(100),                       -- 版本描述
    release_start_date VARCHAR(20),                 -- 生效日期
    release_end_date VARCHAR(20),                   -- 结束日期
    creator VARCHAR(50),                           -- 创建者
    api_timestamp BIGINT,                          -- API返回的时间戳
    check_time TIMESTAMP NOT NULL,                 -- 检查时间
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP -- 记录创建时间
);
```

## 监控逻辑

1. **获取当前版本**：调用 `https://hts.usitc.gov/reststop/releaseList`，查找 `status="current"` 的记录
2. **比较版本**：与数据库中最新记录对比
3. **版本更新处理**：
   - 新版本：发送Lark通知 + 保存到数据库
   - 版本未变：仅更新check_time字段
   - 首次运行：保存初始版本 + 发送通知

## Lark通知格式

```
【HTS版本更新】
新版本: 2026HTSRev1
描述: 2026 HTS Revision 1
生效日期: 01/16/2026
检测时间: 2026-01-20 10:00:00
```

## Crontab管理

### 查看当前定时任务
```bash
crontab -l
```

### 手动添加定时任务
```bash
# 每小时执行
(crontab -l; echo "0 * * * * /path/to/run_monitor.sh") | crontab -
```

### 删除定时任务
```bash
# 删除所有HTS监控任务
crontab -l | grep -v 'run_monitor.sh' | crontab -
```

## 故障排查

### 1. 权限问题
```bash
# 确保脚本可执行
chmod +x run_monitor.sh monitor.py setup_cron.sh
```

### 2. 依赖问题
```bash
# 手动安装依赖
source ../../venv/bin/activate
pip install psycopg2-binary requests
```

### 3. 数据库连接失败
- 检查DATABASE_DSN格式是否正确
- 确认数据库服务器可访问
- 验证用户名密码是否正确

### 4. Lark通知失败
- 确认Webhook URL有效
- 检查网络连接
- 查看日志中的错误信息

## 依赖包

- `psycopg2-binary`: PostgreSQL数据库驱动
- `requests`: HTTP请求库

## Python版本要求

- Python 3.6+

## 注意事项

1. **首次运行**会自动创建数据库表
2. **日志文件**会持续增长，建议定期清理或使用logrotate
3. **虚拟环境**在项目根目录的venv/下
4. **时区问题**：cron使用系统时区，注意服务器时区设置

## 作者

HTS版本监控工具 - 2026年1月