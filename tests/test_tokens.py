from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# 初始化百炼
llm = ChatOpenAI(
    api_key="sk-你的百炼SK",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
    stream=False,  # 非流式，最简单，直接拿到usage_metadata
    temperature=0
)

# 简单提问
msg = [HumanMessage(content="你好")]
resp = llm.invoke(msg)

# 直接读取百炼计费token（无需任何回调、on_llm_end）
usage = resp.usage_metadata
print("模型回复：", resp.content)
print("\n===== 百炼API真实计费token =====")
print(f"input_tokens(prompt): {usage['input_tokens']}")
print(f"output_tokens(completion): {usage['output_tokens']}")
print(f"total_tokens: {usage['total_tokens']}")