"""
evaluator.py

Compute the two objectives of the Sustainable RCMPSP:

  Z1 — Total tardiness penalty (EUR)
       Z1 = sum_p  penalty_per_day_p × max(0, completion_p − release_p − deadline_p)

  Z2 — Total stochastic water quota violations (m³)
       Z2 = sum_p  water_m3_per_day_p × sum_{t ∈ active_construction_days_p} ε_t

       where  Q̄_p = water_m3_per_day_p  (nominal daily quota = maximum allowed),
              Q_p_t = Q̄_p × (1 − ε_t)  (effective quota on day t),
              violation_p_t = Q̄_p − Q_p_t = Q̄_p × ε_t  (when project in construction),
       so     Z2 simplifies to the weighted sum of ε_t over active construction days.

"Active construction day" for project p on day t means at least one Construction-
section activity of p is executing on day t.

This formulation incentivises the scheduler to place construction activities on
days where ε_t is small (quota is close to nominal) and avoid days where ε_t is
large (severe restriction).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set

from .loader import Problem


# ── Objectives dataclass ──────────────────────────────────────────────────────

@dataclass
class Objectives:
    z1: float   # EUR — total tardiness penalty
    z2: float   # m³  — total water quota violation

    def as_tuple(self) -> tuple:
        return (self.z1, self.z2)

    def dominates(self, other: "Objectives") -> bool:
        """Return True if self Pareto-dominates other."""
        return (self.z1 <= other.z1 and self.z2 <= other.z2
                and (self.z1 < other.z1 or self.z2 < other.z2))


# ── Main evaluation function ───────────────────────────────────────────────────

def evaluate(start: List[int], prob: Problem, epsilon: List[float]) -> Objectives:
    """
    Evaluate a schedule.

    Parameters
    ----------
    start : List[int]
        start[i] = scheduled start day for activity i (by global ID).
    prob : Problem
        Loaded problem instance.
    epsilon : List[float]
        ε_t series.  epsilon[t] ∈ [0, 1) for t = 0 … len(epsilon)-1.
        Days beyond the series are treated as ε_t = 0 (no restriction).

    Returns
    -------
    Objectives
        z1 in EUR, z2 in m³.
    """
    acts = prob.activities
    T = len(epsilon)

    z1 = 0.0
    z2 = 0.0

    for pname, pm in prob.projects.items():
        ids = pm.activity_ids

        # ── Z1: tardiness ────────────────────────────────────────────────
        finishes = [
            start[g] + acts[g].duration
            for g in ids
            if not acts[g].is_dummy
        ]
        if finishes:
            completion = max(finishes)
            tardiness = max(0, completion - pm.release_date - pm.deadline)
            z1 += pm.penalty_per_day * tardiness

        # ── Z2: water quota violations ───────────────────────────────────
        # Collect days where at least one construction activity is active
        construction_days: Set[int] = set()
        for g in ids:
            a = acts[g]
            if a.is_construction and a.duration > 0:
                s = start[g]
                for d in range(s, s + a.duration):
                    construction_days.add(d)

        eps_sum = sum(
            epsilon[d] if d < T else 0.0
            for d in construction_days
        )
        z2 += pm.water_m3_per_day * eps_sum

    return Objectives(z1=z1, z2=z2)


# ── Per-project breakdown (for reporting) ─────────────────────────────────────

def project_breakdown(start: List[int], prob: Problem,
                      epsilon: List[float]) -> Dict[str, dict]:
    """Return per-project diagnostics dict (for printing / logging)."""
    acts = prob.activities
    T = len(epsilon)
    result: Dict[str, dict] = {}

    for pname, pm in prob.projects.items():
        ids = pm.activity_ids

        finishes = [start[g] + acts[g].duration for g in ids if not acts[g].is_dummy]
        completion = max(finishes) if finishes else pm.release_date
        tardiness = max(0, completion - pm.release_date - pm.deadline)
        penalty = pm.penalty_per_day * tardiness

        construction_days: Set[int] = set()
        for g in ids:
            a = acts[g]
            if a.is_construction and a.duration > 0:
                s = start[g]
                for d in range(s, s + a.duration):
                    construction_days.add(d)

        eps_sum = sum(epsilon[d] if d < T else 0.0 for d in construction_days)
        water_violation = pm.water_m3_per_day * eps_sum

        result[pname] = {
            "completion": completion,
            "deadline_abs": pm.release_date + pm.deadline,
            "tardiness_days": tardiness,
            "penalty_eur": penalty,
            "n_construction_days": len(construction_days),
            "eps_sum": round(eps_sum, 2),
            "water_violation_m3": round(water_violation, 1),
        }

    return result


# ── Quick self-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from .loader import load
    from .sgs import decode
    import random

    prob = load()
    N = len(prob.activities)
    T = prob.horizon

    # Dummy epsilon: moderate scenario (flat 0.30)
    eps_flat = [0.30] * T

    random.seed(0)
    prio = list(range(N))
    random.shuffle(prio)
    start = decode(prio, prob)
    obj = evaluate(start, prob, eps_flat)
    bd = project_breakdown(start, prob, eps_flat)

    print(f"Z1 = {obj.z1:,.0f} EUR")
    print(f"Z2 = {obj.z2:,.1f} m3")
    for pname, d in bd.items():
        print(f"  {pname}: tardiness={d['tardiness_days']}d "
              f"penalty={d['penalty_eur']:.0f}EUR "
              f"constr_days={d['n_construction_days']} "
              f"water_violation={d['water_violation_m3']:.0f}m3")
