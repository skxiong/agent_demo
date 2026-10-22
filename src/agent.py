#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""主 Agent 相关：系统提示词、工具汇总、构建 deep agent，以及 LangFuse 追踪调用。

对外导出：`agent`（编译后的 deep agent）、`run_agent`（带 LangFuse 追踪的调用）。
"""
import logging
from typing import Optional

from deepagents import create_deep_agent
from langfuse import observe, get_client
from langfuse.langchain import CallbackHandler

from src.llm import llm
from src.tool import BASE_TOOLS
from src.skill import SKILLS, SKILL_TOOLS, build_skill_catalog
from src.subagent import DEEP_SUBAGENTS

# 抑制 OpenTelemetry 网络超时日志
logging.getLogger("opentelemetry").setLevel(logging.CRITICAL)

# 主 Agent 可用的全部工具 = 基础工具 + 技能工具（task 子 Agent 工具由 deepagents 注入）
TOOLS = BASE_TOOLS + SKILL_TOOLS

_SKILL_CATALOG = build_skill_catalog(SKILLS)

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

# ==================== 构建主 Agent（deepagents） ====================

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


# ==================== LangFuse 追踪调用 ====================

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