#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""子 Agent（基于 deepagents 三方件）。

通过 deepagents 的 `SubAgent` 定义子 Agent，`create_deep_agent` 会自动注入
`task(task, subagent_type)` 工具，主 Agent 调用该工具即可开启子 Agent。
"""
from deepagents import SubAgent

from src.tool import BASE_TOOLS

# deepagents 会自动注入一个 task(task, subagent_type) 工具供主 Agent 开启子 Agent
DEEP_SUBAGENTS = [
    SubAgent(
        name="research",
        description=(
            "研究与调研子智能体。当任务可以被拆分为独立的调研/查询子任务，"
            "或某一子任务需要单独处理时，交由该子智能体完成。"
        ),
        system_prompt=(
            "你是一个子智能体（research），负责完成主 Agent 交给你的子任务。"
            "请使用可用工具完成任务，聚焦、简洁，最后给出明确的结论。"
        ),
        tools=BASE_TOOLS,
    ),
]