"""
api_key_manager.py
统一管理 Ollama / OpenAI 类 API Key、base_url、模型名称
其他脚本直接导入类使用
"""
import os


class OllamaConfig:
    """Ollama 配置类，ollama本地一般不需要api_key，保留字段兼容接口"""
    API_KEY: str = os.getenv("OLLAMA_API_KEY", "")
    BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
    MODEL_NAME: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")


class OpenAIConfig:
    """OpenAI / 兼容OpenAI协议的接口（deepseek、通义千问、智谱等）"""
    API_KEY: str = os.getenv("OPENAI_API_KEY", "sk-xxxxxxx")
    BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    MODEL_NAME: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
