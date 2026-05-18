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


async def _vram_free_mb() -> int:
    proc = await asyncio.create_subprocess_exec(
        "nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
        stdout=asyncio.subprocess.PIPE,
    )
    out, _ = await proc.communicate()
    return int(out.decode().strip())


async def _vram_total_mb() -> int:
    proc = await asyncio.create_subprocess_exec(
        "nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits",
        stdout=asyncio.subprocess.PIPE,
    )
    out, _ = await proc.communicate()
    return int(out.decode().strip())


async def ollama_unload() -> None:
    url = f"http://{settings.ollama_host}:{settings.ollama_port}/api/chat"
    payload = {"model": OLLAMA_MODEL, "messages": [], "keep_alive": 0}
    async with httpx.AsyncClient(timeout=30) as client:
        await client.post(url, json=payload)
    ps_url = f"http://{settings.ollama_host}:{settings.ollama_port}/api/ps"
    async with httpx.AsyncClient(timeout=60) as client:
        for _ in range(60):
            await asyncio.sleep(1)
            resp = await client.get(ps_url)
            if not resp.json().get("models"):
                break
    total_mb = await _vram_total_mb()
    for _ in range(30):
        if await _vram_free_mb() >= total_mb * 0.7:
            break
        await asyncio.sleep(1)
