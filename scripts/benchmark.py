
import json
import logging
import sys

from stardust.registry import warm_up

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    dataset = sys.argv[1] if len(sys.argv) > 1 else "hotpotqa"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else None

    log.info("warming up models...")
    warm_up()

    if dataset == "hotpotqa":
        from stardust.benchmark.hotpotqa import run
    elif dataset == "qasper":
        from stardust.benchmark.qasper import run
    elif dataset == "crag":
        from stardust.benchmark.crag import run
    else:
        log.error("unknown dataset: %s. choose from: hotpotqa, qasper, crag", dataset)
        sys.exit(1)

    metrics = run(n=n)
    log.info("final metrics:\n%s", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
