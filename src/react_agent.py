import re
from langchain_core.tools import BaseTool

REACT_PROMPT_TEMPLATE = """
你是ReAct智能助手，可以使用工具解决问题。
可用工具列表：
{tool_descriptions}

严格输出格式：
Thought: 你的思考
Action: 工具名称
Action Input: 工具入参

当得到最终答案时，输出：
Final Answer: 你的最终回答

历史上下文：
{history}

用户问题：{query}
"""


def get_tool_descriptions(tool_list: list[BaseTool]) -> str:
    desc = ""
    for t in tool_list:
        desc += f"- {t.name}: {t.description}\n"
    return desc


def parse_react_output(text: str):
    final_ans_pat = r"Final Answer:\s*(.*)"
    action_pat = r"Action:\s*(.*?)\nAction Input:\s*(.*)"

    final_match = re.search(final_ans_pat, text, re.DOTALL)
    if final_match:
        return {"type": "final", "content": final_match.group(1).strip()}

    act_match = re.search(action_pat, text, re.DOTALL)
    if act_match:
        return {
            "type": "action",
            "tool_name": act_match.group(1).strip(),
            "tool_input": act_match.group(2).strip()
        }
    return {"type": "none", "raw": text}


class HandWriteReActAgent:
    def __init__(self, llm, tools: list[BaseTool], max_iter: int = 5):
        self.llm = llm
        self.tools = tools
        self.tool_map = {t.name: t for t in tools}
        self.max_iter = max_iter

    def run(self, query: str) -> str:
        history = ""
        tool_desc = get_tool_descriptions(self.tools)

        for step in range(self.max_iter):
            prompt = REACT_PROMPT_TEMPLATE.format(
                tool_descriptions=tool_desc,
                history=history,
                query=query
            )
            resp = self.llm.invoke(prompt)
            raw_out = resp.content
            print(f"\n==== Step {step+1} LLM Output ====\n{raw_out}\n")

            parsed = parse_react_output(raw_out)
            if parsed["type"] == "final":
                return parsed["content"]

            if parsed["type"] == "action":
                name = parsed["tool_name"]
                arg = parsed["tool_input"]
                if name not in self.tool_map:
                    obs = f"error: unknown tool {name}"
                else:
                    obs = self.tool_map[name].invoke(arg)
                print(f"Observation: {obs}\n")
                history += f"{raw_out}\nObservation: {obs}\n"
            else:
                history += f"{raw_out}\nObservation: 输出格式错误，请严格遵循格式\n"
        return "达到最大迭代次数，无法得到答案"
