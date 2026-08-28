import os
import requests
from src.llm_factory import load_env_file, find_env_file # 导入src读取.env函数


def chat_request(base_url: str, api_key: str, model_name: str, user_content: str, timeout=120):
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": user_content}],
        "temperature": 0.7
    }
    resp = requests.post(url=url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def test_ollama():
    print("=== Test Ollama ===")
    base_url = os.getenv("OLLAMA_BASE_URL")
    model = os.getenv("OLLAMA_MODEL")
    api_key = os.getenv("OLLAMA_API_KEY", "dummy")

    if not (base_url and model):
        print("skip test_ollama: .env缺少 OLLAMA_BASE_URL / OLLAMA_MODEL")
        return

    res = chat_request(base_url, api_key, model, "你好，请简单介绍自己")
    print(res["choices"][0]["message"]["content"])


def test_openai():
    print("\n=== Test OpenAI ===")
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_MODEL")

    if not (api_key and base_url and model):
        print("skip test_openai: .env缺少 OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL")
        return

    res = chat_request(base_url, api_key, model, "hello world")
    print(res["choices"][0]["message"]["content"])


if __name__ == "__main__":
    env_path = find_env_file()
    load_env_file(env_path)

    test_ollama()
    # test_openai()
