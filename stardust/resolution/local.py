from collections.abc import AsyncGenerator

from stardust.config import LLM_BATCH_TOKEN_LIMIT
from stardust.registry import llm_tokenizer

_PREAMBLE = (
    "Resolve coreference. Output a single JSON object mapping token_id to referent_token_id. "
    "Only include pronouns or nominals that clearly refer to another token. "
    "Do not map predicate nominals or role descriptions.\n"
    "Example: [0] Sarah joined the firm. She became partner.\ntokens: 12:Sarah, 13:firm, 14:She, 15:partner\n"
    "[1] The treaty was signed by France. It came into force.\ntokens: 20:treaty, 21:France, 22:It\n"
    '=> {"14":12,"22":20}\n\n'
)


async def build_batch_prompts(
    rows: AsyncGenerator[tuple[int, str, list[dict]]],
) -> AsyncGenerator[tuple[str, list[tuple[int, list[dict]]]]]:
    tokenizer = llm_tokenizer()
    preamble_tokens = len(tokenizer.encode(_PREAMBLE, add_special_tokens=False))
    budget = LLM_BATCH_TOKEN_LIMIT - preamble_tokens
    sections: list[str] = []
    batch_rows: list[tuple[int, list[dict]]] = []
    batch_tokens = 0
    atom_counter = 0
    async for atom_id, value, tokens in rows:
        if not tokens:
            continue
        leaf = value.rsplit(" | ", 1)[-1] if " | " in value else value
        tokens_str = ", ".join(f"{t['id']}:{t['text']}" for t in tokens)
        section = f"[{atom_counter}] {leaf}\ntokens: {tokens_str}"
        section_tokens = len(tokenizer.encode(section, add_special_tokens=False))
        if sections and batch_tokens + section_tokens > budget:
            yield _PREAMBLE + "\n\n".join(sections), batch_rows
            sections = []
            batch_rows = []
            batch_tokens = 0
            atom_counter = 0
            section = f"[0] {leaf}\ntokens: {tokens_str}"
            section_tokens = len(tokenizer.encode(section, add_special_tokens=False))
        sections.append(section)
        batch_rows.append((atom_id, tokens))
        batch_tokens += section_tokens
        atom_counter += 1
    if sections:
        yield _PREAMBLE + "\n\n".join(sections), batch_rows
