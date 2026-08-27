from src.llm_factory import load_env_file, get_llm
from src.tools import ALL_TOOLS
from src.react_agent import HandWriteReActAgent

if __name__ == "__main__":
    # 加载.env配置
    load_env_file(".env")
    # 获取llm实例
    llm = get_llm()
    # 初始化agent
    agent = HandWriteReActAgent(llm=llm, tools=ALL_TOOLS, max_iter=5)

    question = "现在是什么时间，计算 789 * 22 + 100 等于多少"
    result = agent.run(question)
    print(f"\n>>> Agent最终结果：{result}")
