#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""入口演示：ReAct 智能体 + LangFuse 监控（CLI）。

ReAct 范式：Reasoning + Acting
  - LLM 先推理（Thought），决定是否调用工具（Action）
  - 调用工具后拿到观察结果（Observation），再继续推理
  - 直到 LLM 给出最终答案（Final Answer）

模块拆分（均在 src/ 下）：
  - llm.py      : 加载项目 .env 并构建 LLM（llm）
  - tool.py     : 基础工具与 BASE_TOOLS
  - dangerous.py: run_command 与危险命令拦截、用户确认
  - skill.py    : 从 skill 目录加载 SKILL.md 生成技能工具（SKILLS / SKILL_TOOLS）
  - subagent.py : 基于 deepagents 的子 Agent 定义（DEEP_SUBAGENTS）
  - agent.py    : 系统提示词、构建主 Agent、run_agent（LangFuse 追踪）
  - helloagent.py: 本文件，CLI 入口

功能：
  - Skill：问题与技能 description 相似时先加载技能步骤并按步骤执行
  - 子 Agent：任务可拆分时通过 deepagents 的 task 工具开启子 Agent
  - 危险命令：run_command 命中 rm / chown 等模式时先向用户确认（可同意/拒绝）
  - LangFuse：@observe + CallbackHandler 记录整条链路

环境变量配置（Windows PowerShell），或写入项目根目录（agent_demo）的 .env：
  OPENAI_API_KEY / OPENAI_BASE_URL
  LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL
"""
import os
import sys
import uuid

# 项目根目录（src 的上一级，即 agent_demo）；加入 sys.path 以便导入 src 包
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from langfuse import get_client

from src.agent import run_agent
from src.skill import SKILLS
from src.subagent import DEEP_SUBAGENTS

LANGFUSE_ENABLED = bool(
    os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")
)


def demo_react_agent():
    print("\n" + "=" * 60)
    print("ReAct 智能体演示（LangFuse 追踪）")
    print("=" * 60)
    print("请输入你的问题，ReAct 智能体会调用工具作答。")
    print("输入 exit / quit 或直接回车退出，任何时候按 Ctrl+C 也可退出。")
    if SKILLS:
        print("已加载技能: " + ", ".join(s["name"] for s in SKILLS))
    print("可用子 Agent 类型: " + ", ".join(s["name"] for s in DEEP_SUBAGENTS) + "（deepagents 自动提供 task 工具）")
    print()

    session_id = f"session-{uuid.uuid4().hex[:8]}"
    user_id = "cli_user"

    try:
        while True:
            try:
                question = input("你: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not question or question.lower() in ("exit", "quit"):
                break

            try:
                answer = run_agent(
                    question=question,
                    user_id=user_id,
                    session_id=session_id,
                    tags=["react", "langgraph", "cli"],
                    metadata={"app_version": "1.0.0"},
                )
            except KeyboardInterrupt:
                print("\n[已中断当前对话]")
                break

            print(f"助手: {answer}\n")
    finally:
        if LANGFUSE_ENABLED:
            get_client().flush()
        print("\n对话已结束，再见！")


if __name__ == "__main__":
    print(f"LangFuse: {'已启用' if LANGFUSE_ENABLED else '未配置（跳过追踪）'}")
    demo_react_agent()
    print("\n运行完毕。在 LangFuse 控制台可查看 ReAct 链路的追踪数据。")