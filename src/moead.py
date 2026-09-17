"""
moead.py

MOEA/D — Multi-Objective Evolutionary Algorithm based on Decomposition
for the Sustainable RCMPSP.

Reference: Zhang & Li (2007) "MOEA/D: A Multiobjective Evolutionary Algorithm
Based on Decomposition", IEEE TEVC 11(6):712-731.

Architecture
------------
  Representation : priority vector  — a permutation of [0, N-1].
                   priority[k] = global activity ID of the k-th activity to be
                   scheduled.  Decoded to start times via SGS (sgs.decode).

  Scalarisation  : Tchebycheff (weighted Chebyshev):
                   g_te(x|λ, z*) = max_j { λ_j × |f_j(x) − z_j*| }
                   with adaptive normalisation to handle the EUR/m³ scale gap.

  Crossover      : Order Crossover (OX) — preserves relative order of a
                   contiguous segment from parent 1, fills gaps from parent 2.

  Mutation       : Random swap — swap two random positions with probability
                   mut_rate per position (expected ≈ 2 swaps per individual).

  Local search   : Water-arbitrage operator (Marisa's idea):
                   Identifies construction activities scheduled on high-ε days
                   and swaps them to later positions in the priority vector,
                   biased toward reducing Z2.

Experimental protocol (paper)
  - 30 independent runs × 3 scenarios × 2 metrics (HV, IGD)
  - Wilcoxon signed-rank test against NSGA-II baseline
  - pop_size=100, T_neighbours=20, max_gen=500, cx_rate=1.0,
    mut_rate=2/N, wa_attempts=5
"""

from __future__ import annotations

import random
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .evaluator import Objectives, evaluate
from .loader import Problem
from .sgs import decode


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class MOEADResult:
    """What MOEA/D returns after a full run."""
    pareto_front: List[Objectives]          # non-dominated objective pairs
    pareto_solutions: List[List[int]]       # corresponding priority vectors
    history: List[Tuple[int, float, float]] # (generation, best_z1, best_z2) snapshots
    elapsed_sec: float


# ── Utility: non-dominated sorting ───────────────────────────────────────────

def _is_nondominated(obj: Objectives, front: List[Objectives]) -> bool:
    return all(not f.dominates(obj) for f in front)


def _pareto_filter(objectives: List[Objectives],
                   solutions: List[List[int]]) -> Tuple[List[Objectives], List[List[int]]]:
    """Return only non-dominated (obj, sol) pairs, deduplicated by objective values."""
    front: List[Objectives] = []
    sols: List[List[int]] = []
    seen: set = set()   # (z1, z2) tuples already in the front
    for obj, sol in zip(objectives, solutions):
        key = (obj.z1, obj.z2)
        if key in seen:
            continue
        if _is_nondominated(obj, front):
            keep = [(f, s) for f, s in zip(front, sols) if not obj.dominates(f)]
            front = [k[0] for k in keep]
            sols  = [k[1] for k in keep]
            # Prune seen keys that were just removed
            seen = {(f.z1, f.z2) for f in front}
            front.append(obj)
            sols.append(sol)
            seen.add(key)
    return front, sols


# ── Crossover: Order Crossover (OX) ──────────────────────────────────────────

def _order_crossover(p1: List[int], p2: List[int], rng) -> List[int]:
    """
    Order Crossover for permutations.
    Copies a random segment from p1 directly; fills remaining positions
    in the order they appear in p2.
    """
    N = len(p1)
    a, b = sorted(rng.integers(0, N, size=2))
    if a == b:
        b = min(a + 1, N)

    child = [-1] * N
    child[a:b] = p1[a:b]
    segment_set = set(p1[a:b])

    fill_vals = [x for x in p2 if x not in segment_set]
    fill_idx = 0
    for i in range(N):
        if child[i] == -1:
            child[i] = fill_vals[fill_idx]
            fill_idx += 1
    return child


# ── Mutation: random swap ─────────────────────────────────────────────────────

def _swap_mutation(individual: List[int], mut_rate: float, rng) -> List[int]:
    """
    For each position, with probability mut_rate swap it with a random other position.
    mut_rate = 2/N yields ≈ 2 swaps per individual on average.
    """
    ind = list(individual)
    N = len(ind)
    for i in range(N):
        if rng.random() < mut_rate:
            j = rng.integers(0, N)
            ind[i], ind[j] = ind[j], ind[i]
    return ind


# ── Local search: Water-arbitrage operator ────────────────────────────────────

def _water_arbitrage(
    priority: List[int],
    prob: Problem,
    epsilon: List[float],
    rng,
    n_attempts: int = 5,
) -> Tuple[List[int], List[int]]:
    """
    Domain-specific local search biased toward reducing Z2.

    Strategy (Marisa's water-arbitrage idea):
      1. Decode the current priority to a schedule.
      2. For each project, find the active construction day with the highest ε_t.
      3. Identify construction activities running on that day.
      4. Swap one of them to a later position in the priority vector
         (scheduling it later may shift it off the high-ε day).
      5. Accept the move if Z2 does not increase (also accept ties to escape
         local optima).  Reject and restore otherwise.

    Parameters
    ----------
    priority    : current priority vector (permutation)
    prob        : problem instance
    epsilon     : ε_t series
    rng         : numpy RNG
    n_attempts  : max number of swap attempts

    Returns
    -------
    (improved_priority, improved_start)
    """
    acts = prob.activities
    T    = len(epsilon)
    N    = len(priority)

    best = list(priority)
    best_start = decode(best, prob)

    def _z2_fast(start):
        """Compute Z2 only (skip Z1 for speed inside the local search)."""
        z2 = 0.0
        for pname, pm in prob.projects.items():
            for g in pm.activity_ids:
                a = acts[g]
                if a.is_construction and a.duration > 0:
                    s = start[g]
                    for d in range(s, s + a.duration):
                        if d < T:
                            z2 += pm.water_m3_per_day * epsilon[d]
        return z2

    best_z2 = _z2_fast(best_start)

    # Inverse map: rank[g] = position of activity g in priority ordering
    rank = [0] * N
    for pos, g in enumerate(best):
        rank[g] = pos

    for _ in range(n_attempts):
        # --- Find worst (project, day) pair: highest ε_t among active construction days
        worst_eps  = -1.0
        worst_day  = -1
        worst_proj: Optional[str] = None

        for pname, pm in prob.projects.items():
            for g in pm.activity_ids:
                a = acts[g]
                if a.is_construction and a.duration > 0:
                    s = best_start[g]
                    for d in range(s, s + a.duration):
                        if d < T and epsilon[d] > worst_eps:
                            worst_eps  = epsilon[d]
                            worst_day  = d
                            worst_proj = pname

        if worst_proj is None or worst_eps <= 0:
            break   # nothing to improve

        # --- Collect construction activities of worst_proj running on worst_day
        targets = [
            g for g in prob.projects[worst_proj].activity_ids
            if (acts[g].is_construction and acts[g].duration > 0
                and best_start[g] <= worst_day < best_start[g] + acts[g].duration)
        ]
        if not targets:
            break

        # --- Pick a random target; try to delay it
        g = int(rng.choice(targets))
        pos_g = rank[g]
        if pos_g >= N - 1:
            continue

        # Swap g with a random later position
        swap_pos = int(rng.integers(pos_g + 1, N))
        h = best[swap_pos]

        new_priority = list(best)
        new_priority[pos_g], new_priority[swap_pos] = h, g

        new_start = decode(new_priority, prob)
        new_z2 = _z2_fast(new_start)

        if new_z2 <= best_z2:          # accept equal or better
            best        = new_priority
            best_start  = new_start
            best_z2     = new_z2
            rank[g]     = swap_pos
            rank[h]     = pos_g

    return best, best_start


# ── Weight vector utilities ────────────────────────────────────────────────────

def _uniform_weights(n: int) -> List[Tuple[float, float]]:
    """Generate n uniformly spaced weight vectors on the 2D simplex."""
    return [(i / (n - 1), 1 - i / (n - 1)) for i in range(n)]


def _neighbourhood(weights: List[Tuple[float, float]], T: int) -> List[List[int]]:
    """
    For each weight vector, return indices of its T nearest neighbours
    (including itself) by Euclidean distance.
    """
    n = len(weights)
    W = np.array(weights)
    neighbours: List[List[int]] = []
    for i in range(n):
        dists = np.sum((W - W[i]) ** 2, axis=1)
        neighbours.append(list(np.argsort(dists)[:T]))
    return neighbours


# ── Tchebycheff scalarisation (normalised) ────────────────────────────────────

def _tchebycheff(obj: Objectives, lam: Tuple[float, float],
                 ideal: np.ndarray, nadir: np.ndarray) -> float:
    """
    Normalised Tchebycheff scalarisation.

    Normalises each objective to [0, 1] using the current ideal (best) and
    nadir (worst) reference points so that the weight interpretation is
    consistent across the EUR / m³ scale gap.
    """
    f = np.array([obj.z1, obj.z2])
    scale = nadir - ideal
    # Avoid division by zero when all solutions have the same objective value
    scale = np.where(scale < 1e-9, 1.0, scale)
    f_norm = (f - ideal) / scale
    return float(max(lam[0] * f_norm[0], lam[1] * f_norm[1]))


# ── Main MOEA/D algorithm ─────────────────────────────────────────────────────

def run_moead(
    prob:          Problem,
    epsilon:       List[float],
    pop_size:      int   = 100,
    T_neighbours:  int   = 20,
    max_gen:       int   = 500,
    cx_rate:       float = 1.0,
    mut_rate:      float = None,     # default: 2/N
    wa_attempts:   int   = 5,
    seed:          int   = 0,
    verbose:       bool  = True,
    log_every:     int   = 50,
) -> MOEADResult:
    """
    Run MOEA/D on the given problem instance and ε_t series.

    Parameters
    ----------
    prob          : loaded Problem instance
    epsilon       : ε_t series (from scenario.generate_epsilon)
    pop_size      : number of weight vectors / subproblems (N)
    T_neighbours  : neighbourhood size
    max_gen       : number of generations
    cx_rate       : crossover probability (per parent pair)
    mut_rate      : mutation rate per position (default 2/N)
    wa_attempts   : water-arbitrage local search attempts per offspring
    seed          : RNG seed
    verbose       : print progress
    log_every     : generation interval for progress logging

    Returns
    -------
    MOEADResult
    """
    t0  = time.time()
    N   = len(prob.activities)
    rng = np.random.default_rng(seed)

    if mut_rate is None:
        mut_rate = 2.0 / N

    # ── Weight vectors and neighbourhood ────────────────────────────────
    weights    = _uniform_weights(pop_size)
    neighbours = _neighbourhood(weights, T_neighbours)

    # ── Initialise population ────────────────────────────────────────────
    # Generate pop_size random permutations
    pop: List[List[int]] = []
    for _ in range(pop_size):
        prio = list(range(N))
        rng.shuffle(prio)
        pop.append(prio)

    # Evaluate initial population
    pop_starts: List[List[int]] = [decode(p, prob) for p in pop]
    pop_objs:   List[Objectives] = [evaluate(s, prob, epsilon)
                                     for s in pop_starts]

    # ── Reference points ─────────────────────────────────────────────────
    z1_vals = np.array([o.z1 for o in pop_objs])
    z2_vals = np.array([o.z2 for o in pop_objs])
    ideal = np.array([z1_vals.min(), z2_vals.min()])
    nadir = np.array([z1_vals.max(), z2_vals.max()])

    history: List[Tuple[int, float, float]] = []

    if verbose:
        print(f"MOEA/D start  pop={pop_size} T={T_neighbours} "
              f"gen={max_gen} seed={seed}")
        print(f"  Initial ideal: Z1={ideal[0]:,.0f} EUR  Z2={ideal[1]:,.1f} m3")

    # ── Main loop ────────────────────────────────────────────────────────
    for gen in range(1, max_gen + 1):
        # Iterate over every subproblem in random order
        order = list(range(pop_size))
        rng.shuffle(order)

        for i in order:
            nb = neighbours[i]

            # Select 2 parents from neighbourhood
            p1_idx, p2_idx = rng.choice(nb, size=2, replace=False)

            # Crossover
            if rng.random() < cx_rate:
                child = _order_crossover(pop[p1_idx], pop[p2_idx], rng)
            else:
                child = list(pop[p1_idx])

            # Mutation
            child = _swap_mutation(child, mut_rate, rng)

            # Water-arbitrage local search
            if wa_attempts > 0:
                child, child_start = _water_arbitrage(
                    child, prob, epsilon, rng, n_attempts=wa_attempts
                )
            else:
                child_start = decode(child, prob)

            child_obj = evaluate(child_start, prob, epsilon)

            # Update ideal and nadir
            if child_obj.z1 < ideal[0]:
                ideal[0] = child_obj.z1
            if child_obj.z2 < ideal[1]:
                ideal[1] = child_obj.z2
            if child_obj.z1 > nadir[0]:
                nadir[0] = child_obj.z1
            if child_obj.z2 > nadir[1]:
                nadir[1] = child_obj.z2

            # Update neighbouring subproblems (Tchebycheff)
            for j in nb:
                lam  = weights[j]
                g_child = _tchebycheff(child_obj,   lam, ideal, nadir)
                g_curr  = _tchebycheff(pop_objs[j], lam, ideal, nadir)
                if g_child <= g_curr:
                    pop[j]        = child
                    pop_starts[j] = child_start
                    pop_objs[j]   = child_obj

        # Log
        if verbose and (gen % log_every == 0 or gen == max_gen):
            elapsed = time.time() - t0
            pf, _ = _pareto_filter(pop_objs, pop)
            print(f"  Gen {gen:4d}/{max_gen}  "
                  f"|PF|={len(pf):3d}  "
                  f"ideal=({ideal[0]:,.0f} EUR, {ideal[1]:,.1f} m3)  "
                  f"elapsed={elapsed:.1f}s")

        history.append((gen, float(ideal[0]), float(ideal[1])))

    # ── Extract final Pareto front ────────────────────────────────────────
    pareto_objs, pareto_sols = _pareto_filter(pop_objs, pop)

    # Sort by Z1 ascending
    paired = sorted(zip(pareto_objs, pareto_sols), key=lambda x: x[0].z1)
    pareto_objs = [p[0] for p in paired]
    pareto_sols = [p[1] for p in paired]

    elapsed = time.time() - t0
    if verbose:
        print(f"\nDone in {elapsed:.1f}s — Pareto front size: {len(pareto_objs)}")
        print(f"  Z1 range: {pareto_objs[0].z1:,.0f} — {pareto_objs[-1].z1:,.0f} EUR")
        print(f"  Z2 range: {pareto_objs[-1].z2:,.1f} — {pareto_objs[0].z2:,.1f} m3")

    return MOEADResult(
        pareto_front=pareto_objs,
        pareto_solutions=pareto_sols,
        history=history,
        elapsed_sec=elapsed,
    )


# ── Quick smoke-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from .loader import load
    from .scenario import generate_epsilon

    prob    = load()
    epsilon = generate_epsilon("moderate", prob.horizon, seed=0)

    result = run_moead(
        prob, epsilon,
        pop_size=50, T_neighbours=10,
        max_gen=100, seed=0,
        verbose=True, log_every=20,
    )

    print("\nPareto front (Z1 EUR | Z2 m3):")
    for obj in result.pareto_front:
        print(f"  Z1={obj.z1:>10,.0f} EUR   Z2={obj.z2:>10,.1f} m3")
