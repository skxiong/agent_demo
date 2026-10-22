#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 简单的 ReAct 智能体 + LangFuse 监控演示

ReAct 范式：Reasoning + Acting
  - LLM 先进行推理（Thought），决定是否需要调用工具（Action）
  - 调用工具后拿到观察结果（Observation），再继续推理
  - 直到 LLM 认为可以直接给出最终答案（Final Answer）

本文件用 deepagents 的 create_deep_agent 构建一个 ReAct 智能体（LangGraph 之上），
并通过 LangFuse 记录整条链路：Agent 节点、LLM generation、Tool 调用。

模块拆分：
  - tool.py      : 基础工具（calculator / get_weather / recommend_attractions /
                   recommend_food / word_count）与 BASE_TOOLS
  - dangerous.py : Shell 命令执行 run_command 与危险命令拦截、用户确认
  - subagent.py  : 基于 deepagents 的子 Agent 定义（DEEP_SUBAGENTS）
  - helloagent.py: 本文件，配置 / 技能加载 / 组装 Agent / LangFuse 追踪 / CLI

Skill（技能）支持：
  - 从项目根目录的 skill 目录读取 <skill>/SKILL.md（YAML frontmatter: name/description）
  - 每个技能注册为一个工具，description 用于匹配用户问题
  - 当用户问题与某技能描述相似时，智能体先调用该技能工具获取步骤，
    再严格按步骤依次执行（示例技能 tool_calling 会依次调用
    calculator -> get_weather -> recommend_attractions -> recommend_food -> word_count）

子 Agent（Sub-Agent）支持（基于 deepagents 三方件）：
  - 主 Agent 由 deepagents.create_deep_agent 构建
  - 通过 subagents=[SubAgent(...)] 定义子 Agent，deepagents 自动注入 task 工具
  - 主 Agent 调用 task(description, subagent_type) 即可开启子 Agent
  - 子 Agent 执行链路同样由 LangFuse 记录

危险命令确认（Human-in-the-loop）：
  - run_command 工具执行 Shell 命令
  - 命中 rm / chown / chmod / dd / mkfs / kill / shutdown 等危险模式时，
    先向用户确认（y 同意 / 其它拒绝），用户同意才真正执行

LangFuse 追踪方式：
  - @observe 装饰器：把整次 agent 调用包装成一个 chain span
  - langfuse.langchain.CallbackHandler：自动追踪 LangGraph 内部的
    LLM 调用与工具调用，并挂载到当前 @observe 的 trace 上
  - OpenTelemetry span 属性：注入 user_id / session_id / tags

环境变量配置（Windows PowerShell），或写入项目根目录（agent_demo）的 .env：
  OPENAI_API_KEY / OPENAI_BASE_URL
  LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL
"""
import os
import sys
import uuid
import logging
from typing import Optional

import warnings
warnings.filterwarnings("ignore")

# 修复 langchain 版本兼容性
import langchain
for attr in ('verbose', 'debug', 'llm_cache'):
    if not hasattr(langchain, attr):
        setattr(langchain, attr, False if attr != 'llm_cache' else None)

# 项目根目录（src 的上一级，即 agent_demo）；加入 sys.path 以便导入 src 包
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yaml

# ========== LangChain / LangGraph / deepagents ==========
from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from deepagents import create_deep_agent

# LangFuse 导入
from langfuse import observe, get_client
from langfuse.langchain import CallbackHandler

# 本项目拆分出的模块
from src.tool import BASE_TOOLS
from src.subagent import DEEP_SUBAGENTS

# 从项目根目录的 .env 读取配置到环境变量
_env_path = os.path.join(PROJECT_ROOT, ".env")
if os.path.exists(_env_path):
    with open(_env_path, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _key, _, _val = _line.partition("=")
            _key = _key.strip()
            _val = _val.strip().strip('"').strip("'")
            os.environ.setdefault(_key, _val)

# 抑制 OpenTelemetry 网络超时日志
logging.getLogger("opentelemetry").setLevel(logging.CRITICAL)

# ==================== 全局配置 ====================

LANGFUSE_ENABLED = bool(
    os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")
)

# 使用 ChatOpenAI，可兼容 OpenAI / 通义 / DeepSeek 等 OpenAI‑兼容接口
# model_kwargs 中关闭并行工具调用：保证同一时刻只调用一个工具，
# 使「模型决策 -> 工具调用 -> 工具返回」严格按时间顺序逐一打印
llm = ChatOpenAI(
    model="qwen3.8-27b",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    temperature=0,
    model_kwargs={"parallel_tool_calls": False},
)


# ==================== 工具 ====================
# 基础工具定义在 src/tool.py，危险命令执行在 src/dangerous.py，
# 统一通过 BASE_TOOLS 导入（见上方 import）。


# ==================== 消息打印（严格按时间顺序） ====================

def _print_message(msg, prefix: str = "") -> None:
    """按消息类型打印：模型交互内容 / 模型决策(工具或命令) / 工具调用 / 工具返回。"""
    kind = type(msg).__name__
    if kind == "AIMessage":
        content = getattr(msg, "content", "")
        if content:
            print(f"{prefix}[模型] {content}")
        for call in (getattr(msg, "tool_calls", None) or []):
            name = call["name"]
            args = call.get("args", {})
            # 模型决定要调用的工具 / 要执行的命令
            print(f"{prefix}[模型] 决定调用工具 {name}，参数: {args}")
            print(f"{prefix}[工具调用] {name}  参数: {args}")
    elif kind == "ToolMessage":
        print(f"{prefix}[工具返回] {msg.name}: {msg.content}")


# ==================== Skill 加载（读取项目根目录 skill 目录） ====================

SKILL_DIR = os.path.join(PROJECT_ROOT, "skill")


def _parse_skill_md(text: str):
    """解析 SKILL.md 的 YAML frontmatter 与正文。"""
    meta, body = {}, text.strip()
    stripped = text.lstrip()
    if stripped.startswith("---"):
        _, _, rest = stripped.partition("---")
        front, sep, body_text = rest.partition("---")
        if sep:
            meta = yaml.safe_load(front) or {}
            body = body_text.strip()
    return meta, body


def load_skills(skill_root: str):
    """扫描 skill 目录，读取每个 <skill>/SKILL.md。"""
    skills = []
    if not os.path.isdir(skill_root):
        return skills
    for entry in sorted(os.listdir(skill_root)):
        skill_dir = os.path.join(skill_root, entry)
        md_path = os.path.join(skill_dir, "SKILL.md")
        if os.path.isdir(skill_dir) and os.path.isfile(md_path):
            with open(md_path, encoding="utf-8") as f:
                meta, body = _parse_skill_md(f.read())
            skills.append({
                "name": meta.get("name") or entry,
                "description": (meta.get("description") or "").strip(),
                "body": body,
                "path": md_path,
            })
    return skills


SKILLS = load_skills(SKILL_DIR)


def _make_skill_tool(skill: dict) -> StructuredTool:
    """把一个 skill 变成一个工具：description 用于匹配，调用后返回技能步骤。"""

    def _load(task: str = "") -> str:
        return skill["body"]

    return StructuredTool.from_function(
        func=_load,
        name=f"skill_{skill['name']}",
        description=(
            f"[技能] {skill['description']} "
            "当用户问题与上述描述相似时调用本技能，返回的技能步骤需要严格按顺序执行。"
        ),
    )


SKILL_TOOLS = [_make_skill_tool(s) for s in SKILLS]


# 主 Agent 可用的全部工具 = 基础工具 + 技能工具（task 子 Agent 工具由 deepagents 注入）
TOOLS = BASE_TOOLS + SKILL_TOOLS

_SKILL_CATALOG = "\n".join(
    f"   - {s['name']}（skill_{s['name']}）：{s['description']}" for s in SKILLS
) or "   （当前没有可用技能）"

SYSTEM_PROMPT = (
    "你是一个 ReAct 智能体，必须遵循以下规则：\n"
    "1. 先思考（Thought），再决定是否调用工具（Action）。\n"
    "2. 你的首要任务是匹配「技能（Skill）」。只要用户问题与某个技能的描述相似"
    "（例如涉及该技能覆盖的能力，如数学计算 / 城市天气查询 / 文本字符统计），"
    "就必须先调用对应的 skill_<name> 工具获取技能步骤，然后严格按步骤依次执行。\n"
    f"可用技能：\n{_SKILL_CATALOG}\n"
    "3. 仅当没有任何技能匹配时，才直接调用基础工具：\n"
    "   - 数学计算 -> calculator\n"
    "   - 查询城市天气 / 气温 / 是否下雨 -> get_weather\n"
    "   - 推荐城市旅游景点 -> recommend_attractions\n"
    "   - 推荐城市美食 / 小吃 -> recommend_food\n"
    "   - 统计文本字符数 / 字数 / 长度 -> word_count\n"
    "   - 执行 Shell 命令 -> run_command（危险命令如 rm / chown 会先向用户确认）\n"
    "4. 调用工具后根据返回结果（Observation）继续推理。\n"
    "5. 你还可以开启子 Agent（deepagents 能力）：当任务可拆分为独立子任务、"
    "或某部分需要专门处理时，调用 task(description=\"...\", subagent_type=\"research\") "
    "由子 Agent 完成子任务并返回结论。\n"
    "6. 涉及 rm / chown / chmod / dd / mkfs / kill / shutdown 等危险命令时，"
    "必须通过 run_command 执行，运行时系统会向用户确认，用户可同意或拒绝。\n"
    "7. 当可以确定答案时，输出最终答案（Final Answer）。"
)

# ==================== 构建 ReAct 智能体（deepagents） ====================

# 使用 deepagents 的 create_deep_agent 构建主 Agent：
#   - 自带规划(file/todo)等中间件
#   - 通过 subagents 注入子 Agent，并自动提供 task 工具用于开启子 Agent
agent = create_deep_agent(
    model=llm,
    tools=TOOLS,
    system_prompt=SYSTEM_PROMPT,
    subagents=DEEP_SUBAGENTS,
    name="deep_main_agent",
)


# ==================== LangFuse 追踪（核心） ====================

@observe(name="react_agent_invoke", as_type="chain")
def run_agent(
    question: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    tags: Optional[list] = None,
    metadata: Optional[dict] = None,
) -> str:
    """调用 ReAct 智能体，并通过 LangFuse 记录整条 ReAct 链路。

    使用 stream(values) 逐条消费消息，从而严格按时间顺序打印：
      [模型] Agent 与模型的交互内容
      [模型] 决定调用工具 / 执行命令（模型决策）
      [工具调用] 调用了什么工具
      [工具返回] 工具返回了什么
    """
    # 通过 OpenTelemetry span 属性注入用户上下文
    from opentelemetry import trace as otel_trace
    span = otel_trace.get_current_span()
    if span.is_recording():
        if user_id:
            span.set_attribute("langfuse.user.id", user_id)
        if session_id:
            span.set_attribute("langfuse.session.id", session_id)
        if tags:
            span.set_attribute("langfuse.tags", tags)
    if metadata:
        get_client().update_current_span(metadata=metadata)

    handler = CallbackHandler()
    printed = 0
    final_state = None
    for state in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        config={"callbacks": [handler]},
        stream_mode="values",
    ):
        messages = state["messages"]
        for msg in messages[printed:]:
            _print_message(msg)
        printed = len(messages)
        final_state = state

    return final_state["messages"][-1].content


# ==================== 运行演示 ====================

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