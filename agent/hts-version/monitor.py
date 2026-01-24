#!/usr/bin/env python3
"""
HTS Version Monitor
监控美国HTS版本更新，发现新版本时发送Lark通知并更新数据库
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import psycopg2
import requests
from psycopg2.extras import RealDictCursor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 常量配置
HTS_API_URL = "https://hts.usitc.gov/reststop/releaseList"
LARK_WEBHOOK_URL = "https://open.larksuite.com/open-apis/bot/v2/hook/461e14b8-75ae-49ce-b3c9-73b39d06d659"
USER_AGENT = "Mozilla/5.0 (compatible; HTS-Monitor/1.0)"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_HTS_AGENT = PROJECT_ROOT / "agent" / "basic-hts-agent" / "basic_hts_agent.py"
OTHERCHAPTER_AGENT = PROJECT_ROOT / "agent" / "othercharpter-agent" / "othercharpter.py"


class HTSVersionMonitor:
    """HTS版本监控器"""

    def __init__(self, dsn: str):
        """
        初始化监控器

        Args:
            dsn: PostgreSQL连接字符串
        """
        self.dsn = dsn
        self.conn = None
        self.cursor = None

    def __enter__(self):
        """进入上下文管理器"""
        self.conn = psycopg2.connect(self.dsn)
        self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        self._ensure_table()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文管理器"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            if exc_type is None:
                self.conn.commit()
            else:
                self.conn.rollback()
            self.conn.close()

    def _ensure_table(self):
        """确保数据库表存在"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS hts_version (
            id BIGSERIAL PRIMARY KEY,
            hts_version_name VARCHAR(50) NOT NULL UNIQUE,
            description VARCHAR(100),
            release_start_date VARCHAR(20),
            release_end_date VARCHAR(20),
            creator VARCHAR(50),
            api_timestamp BIGINT,
            check_time TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        self.cursor.execute(create_table_sql)
        self.conn.commit()
        logger.info("数据库表hts_version已就绪")

    def fetch_current_version(self) -> Optional[Dict[str, Any]]:
        """
        从HTS API获取当前版本信息

        Returns:
            当前版本的字典信息，如果失败返回None
        """
        try:
            headers = {"User-Agent": USER_AGENT}
            response = requests.get(HTS_API_URL, headers=headers, timeout=30)
            response.raise_for_status()

            releases = response.json()

            # 查找status为current的记录
            for release in releases:
                if release.get("status") == "current":
                    logger.info(f"获取到当前HTS版本: {release.get('name')}")
                    return release

            logger.warning("未找到status=current的版本记录")
            return None

        except requests.RequestException as e:
            logger.error(f"调用HTS API失败: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"解析API响应失败: {e}")
            return None

    def get_latest_db_version(self) -> Optional[str]:
        """
        获取数据库中最新的版本名称

        Returns:
            最新版本名称，如果没有记录返回None
        """
        query = """
        SELECT hts_version_name
        FROM hts_version
        ORDER BY created_at DESC
        LIMIT 1
        """
        self.cursor.execute(query)
        result = self.cursor.fetchone()

        if result:
            return result['hts_version_name']
        return None

    def send_lark_notification(self, version_info: Dict[str, Any]):
        """
        发送Lark通知

        Args:
            version_info: 版本信息字典
        """
        try:
            message = (
                f"【HTS版本更新】\n"
                f"新版本: {version_info.get('name', 'N/A')}\n"
                f"描述: {version_info.get('description', 'N/A')}\n"
                f"生效日期: {version_info.get('releaseStartDate', 'N/A')}\n"
                f"检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )

            payload = {
                "msg_type": "text",
                "content": {
                    "text": message
                }
            }

            response = requests.post(
                LARK_WEBHOOK_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            response.raise_for_status()
            logger.info("Lark通知发送成功")

        except requests.RequestException as e:
            logger.error(f"发送Lark通知失败: {e}")

    def save_version_to_db(self, version_info: Dict[str, Any]):
        """
        保存版本信息到数据库

        Args:
            version_info: 版本信息字典
        """
        # 提取timestamp
        api_timestamp = None
        if version_info.get('id') and isinstance(version_info['id'], dict):
            api_timestamp = version_info['id'].get('timestamp')

        insert_sql = """
        INSERT INTO hts_version (
            hts_version_name,
            description,
            release_start_date,
            release_end_date,
            creator,
            api_timestamp,
            check_time
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (hts_version_name) DO NOTHING
        """

        self.cursor.execute(insert_sql, (
            version_info.get('name'),
            version_info.get('description'),
            version_info.get('releaseStartDate'),
            version_info.get('releaseEndDate'),
            version_info.get('creator'),
            api_timestamp,
            datetime.now()
        ))
        self.conn.commit()
        logger.info(f"版本信息已保存到数据库: {version_info.get('name')}")

    def _run_agent_script(self, script_path: Path, args: list[str]) -> None:
        if not script_path.exists():
            raise FileNotFoundError(f"脚本不存在: {script_path}")
        env = os.environ.copy()
        env.setdefault("DATABASE_DSN", self.dsn)
        env.setdefault("DATABASE_URL", self.dsn)
        cmd = [sys.executable, str(script_path), *args]
        logger.info("执行脚本: %s", " ".join(cmd))
        result = subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT), check=False, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("脚本执行失败: %s\nstdout:\n%s\nstderr:\n%s", script_path, result.stdout, result.stderr)
            raise RuntimeError(f"Script failed: {script_path}")
        if result.stdout:
            logger.info("脚本输出: %s", result.stdout.strip())

    def _run_update_pipeline(self) -> None:
        self._run_agent_script(BASE_HTS_AGENT, ["--dsn", self.dsn])
        self._run_agent_script(OTHERCHAPTER_AGENT, ["--dsn", self.dsn, "--note", "all"])

    def run(self):
        """执行监控任务"""
        logger.info("开始执行HTS版本监控任务")

        # 1. 获取API当前版本
        current_version = self.fetch_current_version()
        if not current_version:
            logger.error("无法获取当前版本信息，任务终止")
            return

        current_version_name = current_version.get('name')

        # 2. 获取数据库最新版本
        db_latest_version = self.get_latest_db_version()

        # 3. 比较版本
        if db_latest_version is None:
            # 首次运行，直接保存
            logger.info(f"首次运行，记录初始版本: {current_version_name}")
            self.save_version_to_db(current_version)
            # 首次运行也发送通知，让用户知道监控已开始
            self.send_lark_notification(current_version)

        elif current_version_name != db_latest_version:
            # 发现新版本
            logger.info(f"发现新版本! 旧版本: {db_latest_version}, 新版本: {current_version_name}")
            self.send_lark_notification(current_version)
            self.save_version_to_db(current_version)
            self._run_update_pipeline()

        else:
            # 版本未变化
            logger.info(f"版本未变化，当前版本: {current_version_name}")
            # 也更新check_time，记录检查历史
            update_sql = """
            UPDATE hts_version
            SET check_time = %s
            WHERE hts_version_name = %s
            """
            self.cursor.execute(update_sql, (datetime.now(), current_version_name))
            self.conn.commit()

        logger.info("HTS版本监控任务完成")


def main():
    """主函数"""
    # 从环境变量或命令行参数获取数据库连接字符串
    dsn = os.environ.get('DATABASE_DSN')

    # 如果环境变量没有，尝试从命令行参数获取
    if not dsn and len(sys.argv) > 1:
        if sys.argv[1].startswith('--dsn='):
            dsn = sys.argv[1][6:]
        elif len(sys.argv) > 2 and sys.argv[1] == '--dsn':
            dsn = sys.argv[2]

    if not dsn:
        # 使用默认连接字符串（从script.txt中的示例）
        dsn = "postgresql://postgres:Xylx1.t123@34.129.224.77:5432/tariff-simulate"
        logger.warning(f"未提供数据库连接字符串，使用默认值: {dsn[:30]}...")

    try:
        with HTSVersionMonitor(dsn) as monitor:
            monitor.run()
    except Exception as e:
        logger.error(f"监控任务执行失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
