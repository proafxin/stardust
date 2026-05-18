import asyncio
import json
import sys
from pathlib import Path

from sqlalchemy import select

from stardust.db import SessionLocal
from stardust.models import AtomModel

OUT = Path("data/disambiguation_cache.json")


async def main() -> None:
    async with SessionLocal() as session:
        result = await session.execute(
            select(AtomModel.id, AtomModel.record_id, AtomModel.disambiguation)
            .where(AtomModel.disambiguation.is_not(None))
        )
        rows = result.fetchall()

    cache = [{"id": r.id, "record_id": r.record_id, "disambiguation": r.disambiguation} for r in rows]
    OUT.write_text(json.dumps(cache, indent=2))
    print(f"exported {len(cache)} atoms to {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
