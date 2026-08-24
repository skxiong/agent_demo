"""
tests/main.py
"""
import requests
from src.api_key_manager import OllamaConfig, OpenAIConfig


def chat_request(config, user_content: str, timeout=120):
    url = f"{config.BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.API_KEY}"
    }
    payload = {
        "model": config.MODEL_NAME,
        "messages": [{"role": "user", "content": user_content}],
        "temperature": 0.7
    }
    resp = requests.post(url=url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def test_ollama():
    print("=== Test Ollama ===")
    res = chat_request(OllamaConfig, "你好，请简单介绍自己")
    print(res["choices"][0]["message"]["content"])


def test_openai():
    print("\n=== Test OpenAI ===")
    res = chat_request(OpenAIConfig, "hello world")
    print(res["choices"][0]["message"]["content"])


if __name__ == "__main__":
    test_ollama()
    # test_openai()
