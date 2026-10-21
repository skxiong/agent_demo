#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" LangGraph + LangFuse 集成演示
LangFuse 基于 OpenTelemetry 架构，提供两种追踪方式：

方式1（推荐）：@observe 装饰器
  - 最简洁，直接修饰函数
  - 自动追踪函数内的所有 LLM 调用
  - 支持嵌套：被装饰的函数互相调用时，自动形成父子 span

方式2：@observe + OTEL span 属性
  - 适用于需要动态传入 user_id / session_id / tags 的场景
  - 通过 OpenTelemetry span 属性注入用户上下文

环境变量配置（Windows PowerShell）：
  $env:OPENAI_API_KEY="sk-xxx"
  $env:OPENAI_BASE_URL="https://api.openai.com/v1"
  $env:LANGFUSE_PUBLIC_KEY="pk-lf-xxx"
  $env:LANGFUSE_SECRET_KEY="sk-lf-xxx"
  $env:LANGFUSE_BASE_URL="http://172.31.234.9:3000"
"""
import os
import uuid
import logging
from typing import TypedDict, Optional

import warnings
warnings.filterwarnings("ignore")

# 修复 langchain 版本兼容性
import langchain
for attr in ('verbose', 'debug', 'llm_cache'):
    if not hasattr(langchain, attr):
        setattr(langchain, attr, False if attr != 'llm_cache' else None)

# ========== 替换为 langchain_openai ==========
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

# LangFuse 导入
from langfuse import observe, get_client

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

# ====== 方式1：@observe 装饰器（推荐） ======

@observe(as_type="generation")
def call_llm(prompt_text: str, model_name: str = "gpt-3.5-turbo") -> str:
    prompt = ChatPromptTemplate.from_template("{text}")
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"text": prompt_text})
    get_client().update_current_generation(
        model=model_name, input=prompt_text, output=result,
    )
    return result


@observe()
def translate(text: str) -> str:
    return call_llm(f"请将以下内容翻译成英文，只返回译文：\n{text}")


@observe()
def summarize(text: str) -> str:
    return call_llm(f"请用一句话总结以下内容：\n{text}")


@observe()
def qa(question: str) -> str:
    return call_llm("写一篇500字的作文")


@observe(name="multi_tool_agent")
def agent_entry(user_input: str):
    answer = qa(user_input)
    summary = summarize(answer)
    translated = translate(summary)
    return {"answer": answer, "summary": summary, "translated": translated}


# ====== 方式2：@observe + 动态设置 trace 属性 ======

class State(TypedDict):
    question: str
    answer: Optional[str]


def chat_node(state: State) -> dict:
    answer = call_llm(f"请用一句话回答：{state['question']}")
    return {"answer": answer}


def build_graph():
    graph = StateGraph(State)
    graph.add_node("chat", chat_node)
    graph.set_entry_point("chat")
    graph.add_edge("chat", END)
    return graph.compile()


@observe(as_type="chain")
def traced_graph_invoke(question, user_id, session_id, tags=None, metadata=None):
    from opentelemetry import trace as otel_trace
    span = otel_trace.get_current_span()
    if span.is_recording():
        span.set_attribute("langfuse.user.id", user_id)
        span.set_attribute("langfuse.session.id", session_id)
        if tags:
            span.set_attribute("langfuse.tags", tags)
    if metadata:
        get_client().update_current_span(metadata=metadata)
    app = build_graph()
    return app.invoke({"question": question})


# ====== 运行演示 ======

def demo_observe_decorator():
    print("\n" + "=" * 60)
    print("方式1：@observe 装饰器")
    print("=" * 60)
    result = agent_entry("什么是大语言模型？")
    print(f"  回答: {result['answer']}")
    print(f"  摘要: {result['summary']}")
    print(f"  译文: {result['translated']}")


def demo_observe_with_context():
    print("\n" + "=" * 60)
    print("方式2：@observe + 动态上下文")
    print("=" * 60)
    session_id = f"session-{uuid.uuid4().hex[:8]}"
    questions = [
        ("user_001", "什么是 Python？"),
        ("user_001", "它和 Java 有什么区别？"),
        ("user_002", "推荐一个入门编程语言"),
    ]
    for user_id, question in questions:
        print(f"\n  [{user_id}] {question}")
        result = traced_graph_invoke(
            question=question, user_id=user_id,
            session_id=session_id,
            tags=["langgraph", "multi-turn"],
            metadata={"app_version": "1.0.0"},
        )
        print(f"  助手: {result['answer']}")
    if LANGFUSE_ENABLED:
        get_client().flush()


if __name__ == "__main__":
    print(f"LangFuse: {'已启用' if LANGFUSE_ENABLED else '未配置（跳过追踪）'}")
    demo_observe_decorator()
    demo_observe_with_context()
    if LANGFUSE_ENABLED:
        get_client().flush()
    print("\n运行完毕。在 LangFuse 控制台可查看追踪数据。")
