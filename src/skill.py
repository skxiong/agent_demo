#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Skill 加载。

从项目根目录的 skill 目录读取每个 <skill>/SKILL.md（YAML frontmatter:
name / description），并把每个技能注册为一个 LangChain 工具，供 Agent 按
description 匹配后调用。
"""
import os

import yaml
from langchain_core.tools import StructuredTool

# 项目根目录（src 的上一级，即 agent_demo）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKILL_DIR = os.path.join(_PROJECT_ROOT, "skill")


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


def make_skill_tool(skill: dict) -> StructuredTool:
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


def build_skill_catalog(skills) -> str:
    """生成给系统提示词用的技能目录文本。"""
    return "\n".join(
        f"   - {s['name']}（skill_{s['name']}）：{s['description']}" for s in skills
    ) or "   （当前没有可用技能）"


# 模块级加载结果
SKILLS = load_skills(SKILL_DIR)
SKILL_TOOLS = [make_skill_tool(s) for s in SKILLS]