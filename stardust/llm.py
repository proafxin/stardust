from functools import cache

from google import genai
from groq import AsyncGroq, Groq

from stardust.config import settings

GROQ_MODEL = "llama-3.3-70b-versatile"
GEMINI_MODEL = "gemini-1.5-flash"


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
