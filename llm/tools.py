from datetime import datetime
from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """数学计算器，计算数学表达式。输入示例："2*3+10" """
    try:
        return str(eval(expression))
    except Exception as e:
        return f"calc error: {str(e)}"


@tool
def get_current_time() -> str:
    """获取当前系统时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# 导出工具列表，新增工具在这里添加
ALL_TOOLS = [calculator, get_current_time]
