"""
run.py  —  entry point and smoke-test for the Sustainable RCMPSP solver.

Run:
    python run.py

Verifies that loader + SGS produce a feasible schedule on the real portfolio.
"""

import random
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent))

from src.loader import load
from src.sgs import decode, check_feasibility


def project_stats(start, prob):
    """Print per-project completion, tardiness and construction span."""
    acts = prob.activities
    for pname, pm in prob.projects.items():
        ids = pm.activity_ids
        # Completion = latest finish of any non-dummy activity in the project
        finishes = [start[g] + acts[g].duration for g in ids if not acts[g].is_dummy]
        completion = max(finishes) if finishes else pm.release_date
        tardiness = max(0, completion - pm.release_date - pm.deadline)

        # Construction span
        constr_ids = [g for g in ids if acts[g].is_construction and not acts[g].is_dummy]
        if constr_ids:
            c_start = min(start[g] for g in constr_ids)
            c_end   = max(start[g] + acts[g].duration for g in constr_ids)
            c_span  = c_end - c_start
        else:
            c_span = 0

        penalty = tardiness * pm.penalty_per_day
        water   = c_span * pm.water_m3_per_day

        print(f"  {pname}: completion={completion} | "
              f"deadline={pm.release_date + pm.deadline} | "
              f"tardiness={tardiness}d | penalty={penalty}EUR | "
              f"construction_span={c_span}d | water={water}m3")
    print()


def run_test(label: str, priority, prob):
    print(f"=== {label} ===")
    start = decode(priority, prob)
    violations = check_feasibility(start, prob)

    N = len(prob.activities)
    makespan = max(start[i] + prob.activities[i].duration for i in range(N))
    print(f"  Makespan : {makespan} days")

    project_stats(start, prob)

    if violations:
        print(f"  VIOLATIONS ({len(violations)}):")
        for v in violations[:10]:   # cap output
            print(f"    {v}")
        if len(violations) > 10:
            print(f"    ... and {len(violations) - 10} more")
    else:
        print("  Feasibility: OK  [PASS]")
    print()
    return len(violations) == 0


def main():
    prob = load()
    N = len(prob.activities)
    print(f"Problem loaded: {N} activities, {len(prob.resources)} resources")
    print(f"Horizon: {prob.horizon} days\n")

    all_ok = True

    # Test 1: identity priority (topological order — best-case for SGS)
    priority_id = list(range(N))
    ok = run_test("Identity priority (gid order)", priority_id, prob)
    all_ok = all_ok and ok

    # Test 2: reverse identity (stress test)
    priority_rev = list(reversed(range(N)))
    ok = run_test("Reversed priority", priority_rev, prob)
    all_ok = all_ok and ok

    # Test 3: random shuffle (10 seeds)
    for seed in range(10):
        random.seed(seed)
        prio = list(range(N))
        random.shuffle(prio)
        ok = run_test(f"Random priority (seed={seed})", prio, prob)
        all_ok = all_ok and ok

    print("=" * 50)
    if all_ok:
        print("All tests passed — SGS is feasible on all priority vectors  [PASS]")
    else:
        print("Some tests FAILED — check violations above.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
