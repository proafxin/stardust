import json
import logging

from stardust.benchmark.crag import run as run_crag
from stardust.benchmark.hotpotqa import run as run_hotpotqa
from stardust.benchmark.qasper import run as run_qasper

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

if __name__ == "__main__":
    for name, run in [("hotpotqa", run_hotpotqa), ("qasper", run_qasper), ("crag", run_crag)]:
        log.info("running %s benchmark...", name)
        metrics = run()
        log.info("%s metrics:\n%s", name, json.dumps(metrics, indent=2))
