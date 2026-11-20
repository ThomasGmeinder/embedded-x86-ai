"""How do the cube layouts of an earlier run go with other noise seeds of the model?

Run with .venv, PYTHONPATH = <workshop>/src, from the workshop folder:
    earlier_layouts_with_other_noise.py <earlier run folder> <seeds, comma separated> <policy seed bases, comma separated>

The model draws the noise it starts each answer from out of the policy seed.
A random stacking run gives test number i the policy seed <base> + i
(sapien_rollout.run_stack_batch). When the placement rule changes, the
product's runner cannot place an earlier run's layouts any more. Here each
named seed of the earlier run is run with its earlier cube places and, for
every base, with the policy seed that base gives it. Everything else is the
random stacking test of that seed (chosen_layout_tests.py).

Writes vla_testing/random-stacking-earlier-layouts-sapien-<time>/ with the
same files as a random stacking run; a test is in seed-<n>-policy-<m>/.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

from chosen_layout_tests import ChosenTest, mark_complete, run_chosen_layouts, went
from so101_smolvla.sapien_rollout import VLA_TESTING


def main(earlier_run: str, seeds: str, bases: str) -> None:
    earlier = Path(earlier_run)
    rows = json.loads((earlier / "summary.json").read_text())
    before = {row["environment_seed"]: row for row in rows}
    # The earlier run's own base: the policy seed of its first test.
    earlier_base = rows[0]["policy_seed"]
    tests = []
    for base in map(int, bases.split(",")):
        for seed in map(int, seeds.split(",")):
            policy_seed = base + before[seed]["policy_seed"] - earlier_base
            tests.append(ChosenTest(
                name=f"seed-{seed:05d}-policy-{policy_seed:05d}",
                environment_seed=seed,
                policy_seed=policy_seed,
                layout=before[seed]["initial_layout"],
                notes={"policy_seed_base": base,
                       "earlier": went(before[seed]),
                       "earlier_policy_seed": before[seed]["policy_seed"]},
            ))

    destination = VLA_TESTING / (
        f"random-stacking-earlier-layouts-sapien-{datetime.now():%Y%m%d-%H%M%S}")
    rows = run_chosen_layouts(tests, destination)
    mark_complete(destination, rows, earlier_run=earlier.name,
                  policy_seed_bases=[int(base) for base in bases.split(",")])
    print(f"stacked {sum(bool(row['success']) for row in rows)} of {len(rows)} "
          f"in {destination}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
