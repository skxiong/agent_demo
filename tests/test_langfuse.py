#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 简单的 ReAct 智能体 + LangFuse 监控演示

ReAct 范式：Reasoning + Acting
  - LLM 先进行推理（Thought），决定是否需要调用工具（Action）
  - 调用工具后拿到观察结果（Observation），再继续推理
  - 直到 LLM 认为可以直接给出最终答案（Final Answer）

本文件用 LangGraph 的 create_react_agent 构建一个最简 ReAct 智能体，
并通过 LangFuse 记录整条链路：Agent 节点、LLM generation、Tool 调用。

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

# ========== LangChain / LangGraph ==========
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
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
    """计算数学表达式，例如 '2 + 3 * 4'。"""
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as exc:  # noqa: BLE001
        return f"计算失败: {exc}"


@tool
def get_weather(city: str) -> str:
    """查询指定城市的当前天气（示例数据）。"""
    mock = {"北京": "晴，25°C", "上海": "多云，28°C", "深圳": "阵雨，30°C"}
    return mock.get(city, f"{city}：暂无数据（示例）")


@tool
def word_count(text: str) -> str:
    """统计文本的字符数。"""
    return f"字符数: {len(text)}"


TOOLS = [calculator, get_weather, word_count]

SYSTEM_PROMPT = (
    "你是一个 ReAct 智能体。请先思考（Thought），再决定是否需要调用工具（Action）。"
    "可用工具：calculator（数学计算）、get_weather（查询天气）、word_count（统计字符数）。"
    "拿到工具结果（Observation）后继续推理，直到可以直接给出最终答案（Final Answer）。"
)

# ==================== 构建 ReAct 智能体 ====================

agent = create_react_agent(
    llm,
    TOOLS,
    prompt=SYSTEM_PROMPT,
    name="react_agent",
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

    # CallbackHandler 会自动追踪 LangGraph 内部的 LLM 与工具调用
    handler = CallbackHandler()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"callbacks": [handler]},
    )
    return result["messages"][-1].content


# ==================== 运行演示 ====================

def demo_react_agent():
    print("\n" + "=" * 60)
    print("ReAct 智能体演示（LangFuse 追踪）")
    print("=" * 60)
    session_id = f"session-{uuid.uuid4().hex[:8]}"
    questions = [
        ("user_001", "北京现在的天气怎么样？"),
        ("user_001", "帮我算一下 (12 + 8) * 3 等于多少"),
        ("user_002", '"Langfuse" 这个词有多少个字符？'),
    ]
    for user_id, question in questions:
        print(f"\n  [{user_id}] {question}")
        answer = run_agent(
            question=question,
            user_id=user_id,
            session_id=session_id,
            tags=["react", "langgraph"],
            metadata={"app_version": "1.0.0"},
        )
        print(f"  助手: {answer}")


if __name__ == "__main__":
    print(f"LangFuse: {'已启用' if LANGFUSE_ENABLED else '未配置（跳过追踪）'}")
    demo_react_agent()
    if LANGFUSE_ENABLED:
        get_client().flush()
    print("\n运行完毕。在 LangFuse 控制台可查看 ReAct 链路的追踪数据。")