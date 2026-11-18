"""Run real-storage simulations; each spawned worker owns independent seeded worlds."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    started = time.perf_counter()
    from tests.sim.report import DEFAULT_WORKERS, MAX_WORKERS, run_campaign
    from tests.sim.scenarios import SCENARIOS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["sqlite", "postgres"], default="sqlite")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=1000)
    parser.add_argument("--workers", type=int, choices=range(1, MAX_WORKERS + 1), default=DEFAULT_WORKERS)
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--minimize-budget", type=int, default=0, help="extra disposable reruns permitted per failure")
    args = parser.parse_args()
    if args.seeds < 1 or args.minimize_budget < 0:
        parser.error("seeds must be positive and minimization budget nonnegative")
    result = run_campaign(
        backend=args.backend,
        seed=args.seed,
        seeds=args.seeds,
        scenario=args.scenario,
        workers=args.workers,
        replay=json.loads(args.replay.read_text()) if args.replay else None,
        minimize_budget=args.minimize_budget,
    )
    if "error" not in result:
        elapsed = time.perf_counter() - started
        result["elapsed_seconds"] = round(elapsed, 4)
        result["seeds_per_minute"] = round(result["seeds"] / elapsed * 60, 1)
    if args.report:
        args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return int("error" in result)


if __name__ == "__main__":
    raise SystemExit(main())
