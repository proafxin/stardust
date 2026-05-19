import json
import logging

from stardust.llm import ollama_complete

log = logging.getLogger(__name__)

_PREAMBLE = (
    "You are given a list of named entities of the same type. "
    "Group entities that refer to the same real-world entity. "
    "Output a single JSON object where each key is the token_id of the canonical (most complete) surface form "
    "and the value is a list of all token_ids that refer to the same entity (including the canonical one). "
    "Every token_id must appear in exactly one group.\n"
    "Example input: PERSON: 1:Barack Obama, 2:Obama, 3:the president, 4:George Bush, 5:Bush\n"
    'Example output: {"1": [1, 2, 3], "4": [4, 5]}\n\n'
)


async def canonicalize_by_type(
    ent_type: str, tokens: list[dict]
) -> dict[int, list[int]]:
    if not tokens:
        return {}
    tokens_str = ", ".join(f"{t['id']}:{t['text']} ({t['context']})" for t in tokens)
    prompt = _PREAMBLE + f"{ent_type}: {tokens_str}"
    raw = await ollama_complete(prompt, max_tokens=max(500, len(tokens) * 20))
    try:
        start, end = raw.find("{"), raw.rfind("}") + 1
        result = json.loads(raw[start:end]) if start != -1 and end > 0 else {}
        return {int(k): [int(v) for v in vs] for k, vs in result.items() if str(k).lstrip("-").isdigit()}
    except (json.JSONDecodeError, ValueError):
        log.warning("canonicalize_by_type: failed to parse LLM response for %s: %s", ent_type, raw[:200])
        return {}
