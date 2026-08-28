import os
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

def find_env_file():
    """
    查找.env文件
    1. 优先：当前脚本运行工作目录下查找 .env
    2. 没找到：向上一层，项目根目录（tests上级）查找 .env
    返回.env绝对路径；找不到返回None
    """
    # 1. 当前工作目录
    cwd_env = os.path.join(os.getcwd(), ".env")
    if os.path.isfile(cwd_env):
        return cwd_env

    # 2. 当前脚本所在tests目录，往上到项目根目录
    tests_file = os.path.abspath(__file__)
    tests_dir = os.path.dirname(tests_file)
    project_root = os.path.dirname(tests_dir)
    project_env = os.path.join(project_root, "src", ".env")
    if os.path.isfile(project_env):
        return project_env

    return None

def load_env_file(env_path: str = ".env") -> None:
    """手动解析.env，写入进程环境变量，不依赖python‑dotenv"""
    if not os.path.exists(env_path):
        env_path = find_env_file()
    if not os.path.exists(env_path):
        return None
    
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", maxsplit=1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                os.environ[key] = value


def get_llm():
    """
    自动选择模型
    优先 OpenAI：读取 OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL
    无OpenAI key则使用原生Ollama：OLLAMA_MODEL / OLLAMA_BASE_URL
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_base_url = os.getenv("OPENAI_BASE_URL")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

    ollama_model = os.getenv("OLLAMA_MODEL")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

    if openai_api_key:
        print("检测到OPENAI_API_KEY，优先使用 OpenAI 兼容后端")
        init_kwargs = {
            "model": openai_model,
            "temperature": 0,
            "api_key": openai_api_key
        }
        if openai_base_url:
            init_kwargs["base_url"] = openai_base_url
        return ChatOpenAI(**init_kwargs)

    elif ollama_model:
        print("未配置OPENAI_API_KEY，使用原生 Ollama 后端")
        return ChatOllama(
            model=ollama_model,
            temperature=0,
            base_url=ollama_base_url
        )
    else:
        raise RuntimeError(
            "未配置模型！.env 请配置 OPENAI_API_KEY 或者 OLLAMA_MODEL"
        )
