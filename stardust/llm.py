import asyncio

import httpx

from stardust.config import settings

OLLAMA_MODEL = "qwen3:4b"


async def ollama_complete(prompt: str, max_tokens: int = 4096, keep_alive: str = "5m") -> str:
    url = f"http://{settings.ollama_host}:{settings.ollama_port}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"num_predict": max_tokens, "num_ctx": 12288},
        "keep_alive": keep_alive,
    }
    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json()["message"]["content"]


async def ollama_unload() -> None:
    url = f"http://{settings.ollama_host}:{settings.ollama_port}/api/chat"
    payload = {"model": OLLAMA_MODEL, "messages": [], "keep_alive": 0}
    async with httpx.AsyncClient(timeout=30) as client:
        await client.post(url, json=payload)
    ps_url = f"http://{settings.ollama_host}:{settings.ollama_port}/api/ps"
    async with httpx.AsyncClient(timeout=30) as client:
        for _ in range(30):
            await asyncio.sleep(1)
            resp = await client.get(ps_url)
            if not resp.json().get("models"):
                break
