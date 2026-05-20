import json
import logging

from stardust.benchmark.hotpotqa import run as run_hotpotqa

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

if __name__ == "__main__":
    log.info("running hotpotqa benchmark...")
    metrics = run_hotpotqa()
    log.info("hotpotqa metrics:\n%s", json.dumps(metrics, indent=2))
