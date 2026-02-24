"""
sgs.py

Schedule Generation Scheme (SGS) for the Sustainable RCMPSP.

Takes a priority vector (permutation of [0..N-1]) and returns a feasible
schedule (start day per activity) satisfying:
  1. Release dates per project
  2. Precedence constraints (FS+lag, inter-project)
  3. Resource capacity constraints (non-preemptive)

Algorithm (serial SGS):
  1. Topological sort (Kahn's algorithm).
  2. Unconstrained Earliest Start Times (network diagram, FS+lag).
  3. For each activity in priority order (eligible activities only):
       - Find earliest day >= EST where resource constraints hold.
       - Schedule it; update successors' ESTs eagerly.

Performance design:
  - Resource usage is a flat Python list indexed by (day * n_res + r_idx).
    Python list element access is faster than numpy array access for
    individual element reads/writes in tight loops.
  - _earliest_feasible uses a "jump" strategy: when day k is the first
    conflict in [s, s+d), jump directly to k+1 instead of scanning day by day.
  - Each activity stores only the resource indices it actually uses
    (typically 1-4 out of 12), avoiding checks on zero-demand resources.

Priority convention: priority[k] = global activity ID of the k-th activity
to schedule.  Lower position = scheduled sooner.
"""

from __future__ import annotations

import heapq
from typing import Dict, List, Tuple

import numpy as np

from .loader import Problem


# ── Public entry point ────────────────────────────────────────────────────────

def decode(priority: List[int], prob: Problem) -> List[int]:
    """
    Convert a priority vector into a feasible schedule.

    Parameters
    ----------
    priority : List[int]
        Permutation of [0, N-1].  priority[k] = gid scheduled k-th.
    prob : Problem
        Loaded problem instance.

    Returns
    -------
    start : List[int]
        start[i] = scheduled start day for activity i (global ID).
    """
    N = len(prob.activities)
    assert len(priority) == N, f"Priority length {len(priority)} != {N}"

    acts = prob.activities

    # ── Resource setup ─────────────────────────────────────────────────────
    resource_ids = prob.resource_ids()
    n_res = len(resource_ids)
    rid_to_idx: Dict[str, int] = {rid: i for i, rid in enumerate(resource_ids)}
    caps: List[int] = [prob.resource_map[rid].capacity for rid in resource_ids]

    # Per-activity: list of (r_idx, demand, capacity) for non-zero resources.
    # Using tuples inside a list is faster than numpy for individual element access.
    act_res: List[List[Tuple[int, int, int]]] = []
    for a in acts:
        triples = [
            (rid_to_idx[rid], req, caps[rid_to_idx[rid]])
            for rid, req in a.demands.items()
            if req > 0
        ]
        act_res.append(triples)

    # Flat usage array: usage[day * n_res + r_idx] = units used
    horizon = prob.horizon
    usage: List[int] = [0] * (horizon * n_res)

    # ── Core helpers (pure Python, tight inner loops) ──────────────────────
    def _reserve(s: int, d: int, gid: int) -> None:
        res = act_res[gid]
        if not res:
            return
        for day in range(s, s + d):
            base = day * n_res
            for r_idx, req, _ in res:
                usage[base + r_idx] += req

    def _earliest_feasible(s: int, d: int, gid: int) -> int:
        """
        Find earliest day >= s where activity gid fits.
        Jump strategy: on conflict at day k within the window, jump to k+1.
        """
        if d == 0:
            return s
        res = act_res[gid]
        if not res:
            return s
        while True:
            first_conflict = d   # "no conflict" sentinel
            for k in range(d):
                base = (s + k) * n_res
                for r_idx, req, cap in res:
                    if usage[base + r_idx] + req > cap:
                        first_conflict = k
                        break
                if first_conflict < d:
                    break
            if first_conflict == d:
                return s
            s = s + first_conflict + 1

    # ── Rank array ─────────────────────────────────────────────────────────
    rank: List[int] = [0] * N
    for pos, gid in enumerate(priority):
        rank[gid] = pos

    # ── Release dates per activity ─────────────────────────────────────────
    release: List[int] = [prob.projects[a.project].release_date for a in acts]

    # ── Topological sort (Kahn) ────────────────────────────────────────────
    in_deg = [len(a.predecessors) for a in acts]
    q: List[int] = [i for i in range(N) if in_deg[i] == 0]
    topo_order: List[int] = []
    temp_in = list(in_deg)
    while q:
        node = q.pop()
        topo_order.append(node)
        for edge in acts[node].successors:
            temp_in[edge.succ] -= 1
            if temp_in[edge.succ] == 0:
                q.append(edge.succ)
    if len(topo_order) != N:
        raise ValueError("Cycle detected in precedence graph!")

    # ── Unconstrained ESTs (network diagram, FS+lag) ──────────────────────
    est: List[int] = list(release)
    for gid in topo_order:
        dur = acts[gid].duration
        for edge in acts[gid].successors:
            candidate = est[gid] + dur + edge.lag
            if candidate > est[edge.succ]:
                est[edge.succ] = candidate

    # ── Serial SGS scheduling loop ─────────────────────────────────────────
    scheduled: List[bool] = [False] * N
    start: List[int] = [-1] * N
    pending: List[int] = list(in_deg)

    heap: List[tuple] = []
    for i in range(N):
        if pending[i] == 0:
            heapq.heappush(heap, (rank[i], i))

    scheduled_count = 0
    while heap:
        _, gid = heapq.heappop(heap)
        if scheduled[gid]:
            continue

        act = acts[gid]
        dur = act.duration

        s = _earliest_feasible(est[gid], dur, gid)
        start[gid] = s
        scheduled[gid] = True
        scheduled_count += 1
        _reserve(s, dur, gid)

        finish = s + dur
        for edge in act.successors:
            sg = edge.succ
            candidate = finish + edge.lag
            if candidate > est[sg]:
                est[sg] = candidate
            pending[sg] -= 1
            if pending[sg] == 0:
                heapq.heappush(heap, (rank[sg], sg))

    if scheduled_count != N:
        raise ValueError(
            f"SGS scheduled {scheduled_count}/{N} activities — "
            "cycle or disconnected graph."
        )

    return start


# ── Feasibility checker (for testing) ────────────────────────────────────────

def check_feasibility(start: List[int], prob: Problem) -> List[str]:
    """
    Verify a schedule against all constraints.
    Returns a list of violation descriptions (empty = feasible).
    """
    acts = prob.activities
    resource_ids = prob.resource_ids()
    n_res = len(resource_ids)
    rid_to_idx = {rid: i for i, rid in enumerate(resource_ids)}
    caps = np.array(
        [prob.resource_map[rid].capacity for rid in resource_ids], dtype=np.int32
    )
    violations: List[str] = []

    # 1. Release dates
    for a in acts:
        rd = prob.projects[a.project].release_date
        if start[a.gid] < rd:
            violations.append(
                f"Activity {a.gid} ({a.name}) starts {start[a.gid]} "
                f"< release {rd} of {a.project}"
            )

    # 2. Precedences
    for a in acts:
        for edge in a.successors:
            finish_pred = start[edge.pred] + acts[edge.pred].duration
            if start[edge.succ] < finish_pred + edge.lag:
                violations.append(
                    f"Prec violated: {edge.pred}→{edge.succ} lag={edge.lag}: "
                    f"finish_pred={finish_pred} start_succ={start[edge.succ]}"
                )

    # 3. Resource capacity (numpy, done once)
    makespan = max(start[i] + acts[i].duration for i in range(len(acts)))
    usage = np.zeros((makespan + 1, n_res), dtype=np.int32)
    for a in acts:
        s = start[a.gid]
        if a.duration > 0:
            for rid, req in a.demands.items():
                usage[s:s + a.duration, rid_to_idx[rid]] += req

    for day, r_idx in np.argwhere(usage > caps):
        rid = resource_ids[r_idx]
        violations.append(
            f"Resource {rid} overloaded day {day}: "
            f"used={usage[day, r_idx]}, cap={caps[r_idx]}"
        )

    return violations


# ── Quick self-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from .loader import load
    import random, time

    prob = load()
    N = len(prob.activities)

    for label, prio in [
        ("identity", list(range(N))),
        ("reversed", list(reversed(range(N)))),
        ("random42", (lambda: (random.seed(42), random.sample(range(N), N))[1])()),
    ]:
        t0 = time.time()
        s = decode(prio, prob)
        elapsed = time.time() - t0
        v = check_feasibility(s, prob)
        makespan = max(s[i] + prob.activities[i].duration for i in range(N))
        status = "PASS" if not v else f"FAIL ({len(v)} violations)"
        print(f"{label:10s}: makespan={makespan}  [{status}]  {elapsed*1000:.1f}ms")
