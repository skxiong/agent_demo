#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""危险命令拦截与用户确认（Shell 命令执行）。

`run_command` 工具用于执行 Shell 命令；当命令命中 rm / chown / chmod /
dd / mkfs / kill / shutdown 等危险模式时，会先向用户确认：
用户同意（y/yes）才真正执行，拒绝则取消。
"""
import re
import subprocess
from typing import Optional

from langchain_core.tools import tool

# 命中即视为危险命令，需向用户确认后才可执行
DANGEROUS_COMMAND_PATTERNS = [
    r"\brm\b",        # 删除文件/目录
    r"\brmdir\b",
    r"\bchown\b",     # 修改属主
    r"\bchmod\b",     # 修改权限
    r"\bmkfs\b",      # 格式化
    r"\bdd\b",        # 磁盘写入
    r"\bmv\b",        # 移动/覆盖
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bkill\b",
    r"\bkillall\b",
    r"\bformat\b",
    r"\bdel\b",
    r"\breg\s+delete\b",
    r">\s*/dev/",
    r":\(\)\s*\{",    # fork bomb
]


def _match_dangerous_command(command: str) -> Optional[str]:
    """若命令命中危险模式，返回命中的模式，否则返回 None。"""
    for pattern in DANGEROUS_COMMAND_PATTERNS:
        if re.search(pattern, command):
            return pattern
    return None


def _confirm_dangerous_command(command: str, pattern: str) -> bool:
    """危险命令确认：用户输入 y/yes 表示同意执行，其它视为拒绝。"""
    print(f"\n⚠  [危险命令] 检测到疑似危险操作（匹配 {pattern}）：")
    print(f"     {command}")
    try:
        answer = input("     是否执行？同意请输入 y，拒绝直接回车 (y/N): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    approved = answer in ("y", "yes")
    print(f"     [危险命令] 用户{'已同意' if approved else '已拒绝'}执行。")
    return approved


@tool
def run_command(command: str) -> str:
    """执行 Shell 命令并返回输出（stdout/stderr 与退出码）。
    对于危险命令（如 rm、chown、chmod、dd、mkfs、kill、shutdown 等），
    会先向用户确认：用户同意才执行，拒绝则取消。入参 command 为要执行的命令。"""
    pattern = _match_dangerous_command(command)
    if pattern and not _confirm_dangerous_command(command, pattern):
        return f"用户拒绝执行危险命令，已取消: {command}"

    try:
        proc = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=60,
        )
    except subprocess.TimeoutExpired:
        return f"命令执行超时（>60s）: {command}"
    except Exception as exc:  # noqa: BLE001
        return f"命令执行失败: {exc}"

    output = ((proc.stdout or "") + (proc.stderr or "")).strip()
    return f"exit_code={proc.returncode}\n{output[:2000]}"