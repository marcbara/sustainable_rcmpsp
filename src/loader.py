"""
loader.py

Load portfolio.json into clean Python dataclasses.

Global activity IDs are assigned sequentially across projects:
  P1: 0 .. n1-1
  P2: n1 .. n1+n2-1
  P3: n1+n2 .. total-1

This makes it easy to represent solutions as flat priority vectors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, NamedTuple


# ── Edge ──────────────────────────────────────────────────────────────────────

class Edge(NamedTuple):
    """Directed precedence edge: start[succ] >= end[pred] + lag."""
    pred: int   # global activity ID of predecessor
    succ: int   # global activity ID of successor
    lag: int    # minimum gap in days (0 = standard FS)


# ── Activity ──────────────────────────────────────────────────────────────────

@dataclass
class Activity:
    gid: int                    # global ID (unique across all projects)
    local_id: int               # original ID within the project
    project: str                # "P1", "P2", or "P3"
    name: str
    section: str                # "Pre-Construction" | "Construction" | "Post-Construction"
    duration: int               # days (0 = dummy milestone)
    demands: Dict[str, int]     # resource_id → units/day
    successors: List[Edge] = field(default_factory=list)
    predecessors: List[Edge] = field(default_factory=list)

    @property
    def is_dummy(self) -> bool:
        return self.duration == 0

    @property
    def is_construction(self) -> bool:
        return self.section == "Construction"


# ── Resource ──────────────────────────────────────────────────────────────────

@dataclass
class Resource:
    rid: str        # "R1" .. "R12"
    name: str
    capacity: int


# ── ProjectMeta ───────────────────────────────────────────────────────────────

@dataclass
class ProjectMeta:
    name: str                   # "P1", "P2", "P3"
    release_date: int           # days from global origin
    deadline: int               # days from release_date
    penalty_per_day: int        # €/day of tardiness
    water_m3_per_day: int       # m³/day during construction phase
    activity_ids: List[int]     # global IDs of activities in this project (in original order)


# ── Problem ───────────────────────────────────────────────────────────────────

@dataclass
class Problem:
    activities: List[Activity]              # indexed by global ID
    resources: List[Resource]
    projects: Dict[str, ProjectMeta]
    horizon: int                            # safe upper bound on makespan (days)

    # Convenience lookups
    resource_map: Dict[str, Resource] = field(default_factory=dict)

    def __post_init__(self):
        self.resource_map = {r.rid: r for r in self.resources}

    def capacities(self) -> Dict[str, int]:
        return {r.rid: r.capacity for r in self.resources}

    def resource_ids(self) -> List[str]:
        return [r.rid for r in self.resources]


# ── Loader ────────────────────────────────────────────────────────────────────

def load(path: str | Path = None) -> Problem:
    """
    Parse portfolio.json and return a fully wired Problem instance.

    All edges (intra- and inter-project) are stored on both ends:
      activity.successors contains edges where this activity is the predecessor.
      activity.predecessors contains edges where this activity is the successor.
    """
    if path is None:
        path = Path(__file__).parent.parent / "portfolio.json"
    path = Path(path)

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # ── Resources ──────────────────────────────────────────────────────────
    resources = [
        Resource(rid=r["id"], name=r["name"], capacity=r["capacity"])
        for r in data["resources"]
    ]

    # ── First pass: create Activity objects and assign global IDs ───────────
    # Process projects in a fixed order so global IDs are reproducible.
    project_order = sorted(data["projects"].keys())  # ["P1", "P2", "P3"]

    activities: List[Activity] = []   # indexed by global ID
    # Maps (project_name, local_id) → global_id for edge wiring
    gid_map: Dict[tuple, int] = {}

    projects_raw = data["projects"]

    for pname in project_order:
        proj = projects_raw[pname]
        for act_raw in proj["activities"]:
            gid = len(activities)
            key = (pname, act_raw["id"])
            gid_map[key] = gid
            activities.append(Activity(
                gid=gid,
                local_id=act_raw["id"],
                project=pname,
                name=act_raw["name"],
                section=act_raw["section"],
                duration=act_raw["duration"],
                demands=dict(act_raw["demands"]),
            ))

    # ── Second pass: wire edges ─────────────────────────────────────────────
    for pname in project_order:
        proj = projects_raw[pname]
        for act_raw in proj["activities"]:
            pred_gid = gid_map[(pname, act_raw["id"])]
            pred_act = activities[pred_gid]

            # Intra-project successors
            for s in act_raw["successors"]:
                succ_gid = gid_map[(pname, s["id"])]
                succ_act = activities[succ_gid]
                edge = Edge(pred=pred_gid, succ=succ_gid, lag=s["lag"])
                pred_act.successors.append(edge)
                succ_act.predecessors.append(edge)

            # Inter-project predecessors
            # JSON field: external_predecessors on the SUCCESSOR activity
            # Semantics: start[this] >= end[ext_pred] + lag
            for ep in act_raw["external_predecessors"]:
                ext_pred_gid = gid_map[(ep["project"], ep["id"])]
                ext_pred_act = activities[ext_pred_gid]
                succ_gid = pred_gid  # "pred_gid" here is actually the successor
                edge = Edge(pred=ext_pred_gid, succ=succ_gid, lag=ep["lag"])
                ext_pred_act.successors.append(edge)
                activities[succ_gid].predecessors.append(edge)

    # ── ProjectMeta objects ─────────────────────────────────────────────────
    project_metas: Dict[str, ProjectMeta] = {}
    for pname in project_order:
        proj = projects_raw[pname]
        ids = [gid_map[(pname, a["id"])] for a in proj["activities"]]
        project_metas[pname] = ProjectMeta(
            name=pname,
            release_date=proj["release_date"],
            deadline=proj["deadline"],
            penalty_per_day=proj["penalty_per_day"],
            water_m3_per_day=proj["water_m3_per_day"],
            activity_ids=ids,
        )

    # ── Horizon ─────────────────────────────────────────────────────────────
    # Safe upper bound: max(release + deadline) + sum of all durations
    max_deadline = max(
        pm.release_date + pm.deadline for pm in project_metas.values()
    )
    total_duration = sum(a.duration for a in activities)
    horizon = max_deadline + total_duration

    return Problem(
        activities=activities,
        resources=resources,
        projects=project_metas,
        horizon=horizon,
    )


# ── Quick sanity check ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    prob = load()
    print(f"Activities : {len(prob.activities)}")
    print(f"Resources  : {len(prob.resources)}")
    print(f"Horizon    : {prob.horizon} days")
    print()
    for pname, pm in prob.projects.items():
        acts = [prob.activities[g] for g in pm.activity_ids]
        n_construction = sum(1 for a in acts if a.is_construction)
        n_dummy = sum(1 for a in acts if a.is_dummy)
        lag_edges = sum(1 for a in acts for e in a.successors if e.pred == a.gid and e.lag > 0)
        ext_preds = sum(1 for a in acts for e in a.predecessors if e.pred not in pm.activity_ids)
        print(f"{pname}: {len(acts)} activities "
              f"({n_construction} construction, {n_dummy} dummy) | "
              f"release={pm.release_date} deadline={pm.deadline} "
              f"penalty={pm.penalty_per_day}€/day water={pm.water_m3_per_day}m³/day")
        print(f"  FS+lag successor edges   : {lag_edges}")
        print(f"  Inter-project pred edges : {ext_preds}")
    print()
    # Check edge count
    total_succ = sum(len(a.successors) for a in prob.activities)
    total_pred = sum(len(a.predecessors) for a in prob.activities)
    assert total_succ == total_pred, "Edge mismatch!"
    print(f"Total edges (succ==pred): {total_succ}  ✓")
