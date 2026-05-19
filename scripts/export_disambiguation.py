import asyncio
import json
from pathlib import Path

from sqlalchemy import select

from stardust.db import SessionLocal
from stardust.models import Atom

OUT = Path("data/disambiguation_cache.json")


async def main() -> None:
    async with SessionLocal() as session:
        result = await session.execute(
            select(Atom.id, Atom.record_id, Atom.disambiguation).where(
                Atom.disambiguation.is_not(None)
            )
        )
        rows = result.fetchall()

    cache = [{"id": r.id, "record_id": r.record_id, "disambiguation": r.disambiguation} for r in rows]
    OUT.write_text(json.dumps(cache, indent=2), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
