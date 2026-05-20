import asyncio
import gc
import json
import logging
from pathlib import Path

import torch
from sqlalchemy import select

from stardust.db import SessionLocal
from stardust.models import Sentence
from stardust.registry import embedder as load_embedder
from stardust.registry import unload_embedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CONSERVATISM = 0.90
CONFIG_PATH = Path("tuning.json")


async def get_all_words() -> list[str]:
    async with SessionLocal() as session:
        rows = (await session.execute(select(Sentence.resolved_text))).fetchall()
    return " ".join(r.resolved_text for r in rows).split()


def try_batch(embedder, text: str) -> bool:
    try:
        with torch.no_grad():
            embedder.encode([text], batch_size=1, normalize_embeddings=True, show_progress_bar=False)
        torch.cuda.empty_cache()
        return True
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()
        return False


async def main() -> None:
    unload_embedder()
    gc.collect()
    torch.cuda.empty_cache()

    log.info("total VRAM: %.2fGB", torch.cuda.get_device_properties(0).total_memory / 1024**3)

    embedder = load_embedder()
    tokenizer = embedder.tokenizer

    words = await get_all_words()
    log.info("%d words pooled", len(words))

    cumulative_tokens = []
    total = 0
    for word in words:
        total += len(tokenizer.encode(word, add_special_tokens=False))
        cumulative_tokens.append(total)

    lo, hi, result = 0, len(words) - 1, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if try_batch(embedder, " ".join(words[: mid + 1])):
            result = cumulative_tokens[mid]
            lo = mid + 1
        else:
            hi = mid - 1

    safe_tokens = int(result * CONSERVATISM)
    log.info("max_tokens=%d safe_tokens=%d", result, safe_tokens)

    config: dict = {}
    if CONFIG_PATH.exists():
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["embedding_token_budget"] = safe_tokens
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")
    log.info("written to %s", CONFIG_PATH)


if __name__ == "__main__":
    asyncio.run(main())
