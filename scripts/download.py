import logging
import re
from pathlib import Path

import pyarrow.parquet as pq
from datasets import load_dataset
from pylatexenc.latex2text import LatexNodes2Text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path("data")
HOTPOTQA_DIR = DATA_DIR / "hotpotqa"
QASPER_DIR = DATA_DIR / "qasper"
CRAG_DIR = DATA_DIR / "crag"

QASPER_SPLITS = ["train", "validation"]
CRAG_OPEN_SPLITS = ["train", "test", "sample", "holdout"]

_latex = LatexNodes2Text()


def _clean_qasper(text: str) -> str:
    text = _latex.latex_to_text(text)
    text = re.sub(r"\bBIBREF\d+\b", "", text)
    text = re.sub(r"\bFIGREF\d+\b", "", text)
    text = re.sub(r"\bTABREF\d+\b", "", text)
    text = re.sub(r"\bSECREF\d+\b", "", text)
    text = re.sub(r"\bEQREF\d+\b", "", text)
    return re.sub(r" {2,}", " ", text).strip()


def download_hotpotqa() -> None:
    log.info("converting hotpotqa jsonl to parquet")
    for name in ("corpus", "queries"):
        src = HOTPOTQA_DIR / f"{name}.jsonl"
        out = HOTPOTQA_DIR / f"{name}.parquet"
        if out.exists():
            log.info("hotpotqa %s: already cached", name)
            continue
        if not src.exists():
            log.warning("hotpotqa %s: source jsonl not found, skipping", name)
            continue
        ds = load_dataset("json", data_files=str(src), split="train")  # nosec B615
        ds.to_parquet(out)
        src.unlink()
        log.info("hotpotqa %s: saved %d records", name, len(ds))


def download_qasper() -> None:
    log.info("downloading qasper")
    QASPER_DIR.mkdir(parents=True, exist_ok=True)
    for split in QASPER_SPLITS:
        out = QASPER_DIR / f"{split}.parquet"
        if out.exists():
            log.info("qasper %s: already cached", split)
            continue
        ds = load_dataset("hulki/allenai_qasper", split=split)  # nosec B615
        ds = ds.map(lambda r: {"context": _clean_qasper(r["context"])}, num_proc=4)
        ds.to_parquet(out)
        log.info("qasper %s: saved %d records", split, len(ds))


def download_crag_open() -> None:
    log.info("downloading crag open")
    (CRAG_DIR / "open").mkdir(parents=True, exist_ok=True)
    for split in CRAG_OPEN_SPLITS:
        out = CRAG_DIR / "open" / f"{split}.parquet"
        if out.exists():
            log.info("crag open %s: already cached", split)
            continue
        ds = load_dataset("DataRobot-Research/crag", "groundingdata_open", split=split)  # nosec B615
        ds.to_parquet(out)
        log.info("crag open %s: saved %d records", split, len(ds))


def main() -> None:
    download_hotpotqa()
    download_qasper()
    download_crag_open()


if __name__ == "__main__":
    main()
