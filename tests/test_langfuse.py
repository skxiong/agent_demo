#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 简单的 ReAct 智能体 + LangFuse 监控演示

ReAct 范式：Reasoning + Acting
  - LLM 先进行推理（Thought），决定是否需要调用工具（Action）
  - 调用工具后拿到观察结果（Observation），再继续推理
  - 直到 LLM 认为可以直接给出最终答案（Final Answer）

本文件用 LangGraph 的 create_react_agent 构建一个最简 ReAct 智能体，
并通过 LangFuse 记录整条链路：Agent 节点、LLM generation、Tool 调用。

Skill（技能）支持：
  - 从同级 skill 目录读取 <skill>/SKILL.md（YAML frontmatter: name/description）
  - 每个技能注册为一个工具，description 用于匹配用户问题
  - 当用户问题与某技能描述相似时，智能体先调用该技能工具获取步骤，
    再严格按步骤依次执行（示例技能 tool_calling 会依次调用
    calculator -> get_weather -> word_count）

LangFuse 追踪方式：
  - @observe 装饰器：把整次 agent 调用包装成一个 chain span
  - langfuse.langchain.CallbackHandler：自动追踪 LangGraph 内部的
    LLM 调用与工具调用，并挂载到当前 @observe 的 trace 上
  - OpenTelemetry span 属性：注入 user_id / session_id / tags

环境变量配置（Windows PowerShell），或写入同目录 .env：
  OPENAI_API_KEY / OPENAI_BASE_URL
  LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL
"""
import os
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

import yaml

# ========== LangChain / LangGraph ==========
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, StructuredTool
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.prebuilt import create_react_agent

# LangFuse 导入
from langfuse import observe, get_client
from langfuse.langchain import CallbackHandler

# 从当前文件同级目录的 .env 读取配置到环境变量
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
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
llm = ChatOpenAI(
    model="qwen3.8-27b",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    temperature=0,
)


# ==================== 工具（Acting） ====================

@tool
def calculator(expression: str) -> str:
    """数学计算工具。当用户需要做加、减、乘、除、括号等算术或数值计算时使用。
    入参 expression 为数学表达式字符串，例如 '(12 + 8) * 3'。"""
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as exc:  # noqa: BLE001
        return f"计算失败: {exc}"


@tool
def get_weather(city: str) -> str:
    """天气查询工具。当用户询问某个城市当前的天气、气温、是否下雨等时使用。
    入参 city 为城市名，例如 '北京'。"""
    mock = {"北京": "晴，25°C", "上海": "多云，28°C", "深圳": "阵雨，30°C"}
    return mock.get(city, f"{city}：暂无数据（示例）")


@tool
def word_count(text: str) -> str:
    """字符统计工具。当用户询问某段文本的字符数、字数或长度时使用。
    入参 text 为待统计的文本。"""
    return f"字符数: {len(text)}"


TOOLS = [calculator, get_weather, word_count]


# ==================== Skill 加载（读取同级 skill 目录） ====================

SKILL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skill")


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

# 智能体可用的全部工具 = 基础工具 + 技能工具
TOOLS = TOOLS + SKILL_TOOLS

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
    "   - 统计文本字符数 / 字数 / 长度 -> word_count\n"
    "4. 调用工具后根据返回结果（Observation）继续推理。\n"
    "5. 当可以确定答案时，输出最终答案（Final Answer）。"
)

# ==================== 构建 ReAct 智能体 ====================

agent = create_react_agent(
    llm,
    TOOLS,
    prompt=SYSTEM_PROMPT,
    name="react_agent",
)


# ==================== LangFuse 追踪（核心） ====================

class ToolPrintHandler(BaseCallbackHandler):
    """把智能体实际调用的工具（含技能工具）打印到控制台。"""

    def on_tool_start(self, serialized, input_str, **kwargs):
        name = (serialized or {}).get("name", "?")
        print(f"  [工具调用] {name}  参数: {input_str}")

    def on_tool_end(self, output, **kwargs):
        text = getattr(output, "content", output)
        print(f"  [工具返回] {text}")


@observe(name="react_agent_invoke", as_type="chain")
def run_agent(
    question: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    tags: Optional[list] = None,
    metadata: Optional[dict] = None,
) -> str:
    """调用 ReAct 智能体，并通过 LangFuse 记录整条 ReAct 链路。"""
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

    # CallbackHandler 自动追踪 LLM/工具调用；ToolPrintHandler 打印每次工具调用
    handler = CallbackHandler()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"callbacks": [handler, ToolPrintHandler()]},
    )
    return result["messages"][-1].content


# ==================== 运行演示 ====================

def demo_react_agent():
    print("\n" + "=" * 60)
    print("ReAct 智能体演示（LangFuse 追踪）")
    print("=" * 60)
    print("请输入你的问题，ReAct 智能体会调用工具作答。")
    print("输入 exit / quit 或直接回车退出，任何时候按 Ctrl+C 也可退出。")
    if SKILLS:
        print("已加载技能: " + ", ".join(s["name"] for s in SKILLS))
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