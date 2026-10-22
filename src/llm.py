#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""大模型相关：加载项目根目录 .env，并构建 LLM 实例。

对外导出：`llm`（ChatOpenAI 实例）。
"""
import os
import warnings

warnings.filterwarnings("ignore")

# 修复 langchain 版本兼容性
import langchain
for attr in ('verbose', 'debug', 'llm_cache'):
    if not hasattr(langchain, attr):
        setattr(langchain, attr, False if attr != 'llm_cache' else None)

from langchain_openai import ChatOpenAI

# 项目根目录（src 的上一级，即 agent_demo）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 从项目根目录的 .env 读取配置到环境变量
_env_path = os.path.join(PROJECT_ROOT, ".env")
if os.path.exists(_env_path):
    with open(_env_path, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _key, _, _val = _line.partition("=")
            os.environ.setdefault(_key.strip(), _val.strip().strip('"').strip("'"))

# 使用 ChatOpenAI，可兼容 OpenAI / 通义 / DeepSeek 等 OpenAI 兼容接口
# 模型从 .env 的 OPENAI_MODEL 读取（默认 qwen3.8-flash）
# model_kwargs 关闭并行工具调用：保证「模型决策 -> 工具调用 -> 工具返回」严格按时间顺序
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "qwen3.8-flash"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    temperature=0,
    model_kwargs={"parallel_tool_calls": False},
)