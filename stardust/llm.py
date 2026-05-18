import asyncio
import logging

import httpx

from stardust.config import settings

log = logging.getLogger(__name__)
OLLAMA_MODEL = "qwen3:4b"


async def ollama_complete(prompt: str, max_tokens: int = 4096, keep_alive: str = "5m") -> str:
    url = f"http://{settings.ollama_host}:{settings.ollama_port}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": "You output only valid JSON. No explanation, no markdown, no reasoning."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "format": {"type": "array", "items": {"type": "object"}},
        "options": {"num_predict": max_tokens, "num_ctx": 30000},
        "think": False,
        "keep_alive": keep_alive,
    }
    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json()["message"]["content"]


async def ollama_unload() -> None:
    proc = await asyncio.create_subprocess_exec(
        "ollama", "stop", OLLAMA_MODEL,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.wait()
    log.info("ollama model %s unloaded", OLLAMA_MODEL)
