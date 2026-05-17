from functools import cache

import httpx
from google import genai
from groq import AsyncGroq, Groq

from stardust.config import settings

GROQ_MODEL = "llama-3.3-70b-versatile"
GEMINI_MODEL = "gemini-3.1-flash-lite"
OLLAMA_MODEL = "qwen3:4b"


@cache
def _groq() -> Groq:
    return Groq(api_key=settings.groq_api_key)


@cache
def _async_groq() -> AsyncGroq:
    return AsyncGroq(api_key=settings.groq_api_key)


@cache
def _gemini() -> genai.Client:
    return genai.Client(api_key=settings.gemini_api_key)


def groq_complete(prompt: str, max_tokens: int = 4096) -> str:
    response = _groq().chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


async def async_groq_complete(prompt: str, max_tokens: int = 4096) -> str:
    response = await _async_groq().chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


async def async_gemini_complete(prompt: str, max_tokens: int = 4096) -> str:
    response = await _gemini().aio.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=genai.types.GenerateContentConfig(max_output_tokens=max_tokens),
    )
    return response.text or ""


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
