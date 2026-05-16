import json
import logging

from stardust.extract import extract
from stardust.llm import async_groq_complete
from stardust.parse import Document, build_atom_index
from stardust.resolution.global_resolution import CanonicalEntity, merge_across_documents
from stardust.resolution.local import build_pronoun_prompt, resolve_local
from stardust.query import embed_atoms
from stardust.tree.atom import AtomIndex

log = logging.getLogger(__name__)


def _parse_pronoun_map(raw: str) -> list[dict]:
    try:
        start, end = raw.find("["), raw.rfind("]") + 1
        return json.loads(raw[start:end]) if start != -1 and end > 0 else []
    except json.JSONDecodeError:
        return []


async def index_documents(
    docs: list[Document],
    doc_id: str,
    all_doc_clusters: list[tuple[str, dict]] | None = None,
) -> tuple[AtomIndex, list[CanonicalEntity]]:
    index = build_atom_index(docs)
    log.info("built atom index: %d atoms, %d nodes", len(index.atoms), len(index.nodes))

    extract(index)
    log.info("nlp extraction complete")

    prompt = build_pronoun_prompt(index)
    pronoun_map = _parse_pronoun_map(await async_groq_complete(prompt)) if prompt.strip() else []
    log.info("pronoun resolution: %d mappings", len(pronoun_map))

    clusters = resolve_local(index, pronoun_map)
    log.info("local resolution: %d entity types", len(clusters))

    per_doc = (all_doc_clusters or []) + [(doc_id, clusters)]
    global_entities = merge_across_documents(per_doc)
    log.info("global resolution: %d canonical entities", len(global_entities))

    embed_atoms(index)
    log.info("embeddings complete")

    return index, global_entities
