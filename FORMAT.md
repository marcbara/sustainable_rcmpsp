# Sustainable-RCMPSP JSON Format — v1.0

## Purpose

This document specifies the input format for the **Sustainable Resource-Constrained
Multi-Project Scheduling Problem (RCMPSP)** used in Sánchez et al. (2023).

The format is inspired by the logical structure of standard academic benchmarks
(PSPLIB `.sm`, MPLIB) but extends them to handle three features that those formats
cannot express:

1. **Finish-to-Start precedences with lag** (FS+lag)
2. **Inter-project (cross-project) dependencies**
3. **Phase tags** — distinguishing which activities belong to the construction
   phase, needed to compute the water-consumption objective

---

## Top-Level Structure

```json
{
  "meta":      { ... },
  "resources": [ ... ],
  "projects":  { "P1": { ... }, "P2": { ... }, ... }
}
```

---

## `meta` Object

| Field             | Type   | Description |
|-------------------|--------|-------------|
| `format_version`  | string | Format version (`"1.0"`) |
| `name`            | string | Human-readable instance name |
| `source`          | string | Origin file / citation |
| `num_projects`    | int    | Number of projects |
| `num_resources`   | int    | Number of shared renewable resources |
| `total_activities`| int    | Total activities across all projects (including dummies) |
| `description`     | string | Free-text description of the instance and objectives |

---

## `resources` Array

Each element describes one shared renewable resource.

```json
{
  "id":       "R1",
  "name":     "Architect Sr",
  "type":     "renewable",
  "capacity": 3
}
```

| Field      | Type   | Description |
|------------|--------|-------------|
| `id`       | string | Resource identifier (e.g. `"R1"`) |
| `name`     | string | Human-readable name |
| `type`     | string | Always `"renewable"` in this version |
| `capacity` | int    | Maximum simultaneous units available across all projects |

Resources are **shared**: the combined demand of all concurrently executing
activities across all projects must not exceed `capacity` at any time unit.

---

## `projects` Dictionary

Keys are project names (`"P1"`, `"P2"`, `"P3"`, ...).

```json
"P1": {
  "release_date":     0,
  "deadline":         329,
  "penalty_per_day":  500,
  "water_m3_per_day": 250,
  "activities":       [ ... ]
}
```

| Field             | Type  | Description |
|-------------------|-------|-------------|
| `release_date`    | int   | Earliest start day for any activity in this project (days from global origin) |
| `deadline`        | int   | Contractual deadline in days from `release_date`. Completion beyond this triggers a penalty. |
| `penalty_per_day` | int   | Tardiness penalty in €/day beyond the deadline |
| `water_m3_per_day`| int   | Water consumption rate in m³/day during the project's construction phase |

**Project completion** is defined as the finish of its last non-dummy activity.

**Tardiness** = max(0, completion_day − release_date − deadline)

**Water consumed** = construction_phase_duration × water_m3_per_day, where
`construction_phase_duration` = (latest end of any Construction activity) −
(earliest start of any Construction activity).

---

## Activity Object

```json
{
  "id":       14,
  "name":     "Pipeline system zone 1",
  "section":  "Construction",
  "duration": 7,
  "demands": {
    "R1": 0, "R2": 0, "R3": 0, "R4": 0,
    "R5": 0, "R6": 0, "R7": 0, "R8": 0,
    "R9": 0, "R10": 0, "R11": 0, "R12": 1
  },
  "successors": [
    { "id": 15, "lag": 0 },
    { "id": 23, "lag": 0 }
  ],
  "external_predecessors": [
    { "project": "P2", "id": 9, "lag": 0 },
    { "project": "P3", "id": 8, "lag": 0 }
  ]
}
```

| Field                  | Type   | Description |
|------------------------|--------|-------------|
| `id`                   | int    | Activity identifier, unique within its project |
| `name`                 | string | Human-readable name |
| `section`              | string | Phase tag — see [Section Values](#section-values) |
| `duration`             | int    | Duration in days (0 for dummy milestones) |
| `demands`              | object | Units of each resource required per day during execution |
| `successors`           | array  | Intra-project successors with optional lag — see [Precedences](#precedences) |
| `external_predecessors`| array  | Cross-project predecessors — see [Inter-Project Dependencies](#inter-project-dependencies) |

Activities are **non-preemptive**: once started they run continuously for
`duration` days.

---

## Section Values

| Value               | Meaning |
|---------------------|---------|
| `"Pre-Construction"`| Planning, design, permits, procurement |
| `"Construction"`    | Physical construction works |
| `"Post-Construction"`| Closeout, reporting, acceptance |

The **water-consumption objective** is computed over the span of all
`"Construction"` activities within each project. Only `section == "Construction"`
activities contribute.

**Dummy activities** (duration = 0) at the start and end of each project also
carry a section tag; they do not affect either objective.

---

## Precedences

### Intra-project successors

Each entry in `successors` defines a **Finish-to-Start** constraint:

```
start[successor] >= end[this_activity] + lag
```

```json
"successors": [
  { "id": 26, "lag": 5 }   // start[26] >= end[this] + 5
]
```

| Field | Type | Description |
|-------|------|-------------|
| `id`  | int  | ID of the successor activity (within the same project) |
| `lag` | int  | Minimum gap in days between finish and successor start (0 = standard FS) |

A `lag` of 0 is a standard FS-0 precedence as in PSPLIB/MPLIB.
A positive `lag` is a **FS+lag** constraint (e.g. curing time before painting).

### Inter-Project Dependencies

```json
"external_predecessors": [
  { "project": "P2", "id": 9, "lag": 0 }
]
```

This means: `start[this_activity] >= end[P2.activity_9] + lag`.

| Field     | Type   | Description |
|-----------|--------|-------------|
| `project` | string | Name of the predecessor's project |
| `id`      | int    | ID of the predecessor activity in that project |
| `lag`     | int    | Minimum gap (always 0 in the current instance) |

Inter-project precedences express physical site dependencies — for example,
a shared site cannot be used by two projects simultaneously until one has
completed its groundwork.

---

## Conventions

- **Day indexing**: all times are in integer days from the global origin (day 0 =
  release date of the earliest project).
- **Activity IDs** are local to each project. Activity 14 in P1 and activity 14
  in P2 are different activities. Use `(project, id)` as the global key.
- **Dummy activities** have `duration = 0` and all-zero demands. The first
  activity of each project (`id = 2` in the current instance) is a dummy start
  (contract signing); the last is a dummy end (acceptance milestone).
- **Resource demands** of 0 mean the activity uses none of that resource.
  Activities with all-zero demands are pure precedence nodes.
- **Horizon**: a safe upper bound is
  `max_over_projects(release_date + deadline) + sum_of_all_durations`.

---

## Objectives

This is a **bi-objective minimisation** problem:

| Objective | Formula |
|-----------|---------|
| **O1** — Total penalty | Σ_p  penalty_per_day_p × max(0, completion_p − release_p − deadline_p) |
| **O2** — Total water   | Σ_p  water_m3_per_day_p × (max_end_construction_p − min_start_construction_p) |

where *p* ranges over all projects, and *construction* activities are those
with `section == "Construction"`.

---

## Minimal Parsing Example (Python)

```python
import json

with open("portfolio.json", encoding="utf-8") as f:
    data = json.load(f)

resources    = {r["id"]: r["capacity"] for r in data["resources"]}
projects     = data["projects"]

for pname, proj in projects.items():
    for act in proj["activities"]:
        # Intra-project FS constraints
        for succ in act["successors"]:
            # start[pname, succ["id"]] >= end[pname, act["id"]] + succ["lag"]
            pass
        # Inter-project FS constraints
        for ext in act["external_predecessors"]:
            # start[pname, act["id"]] >= end[ext["project"], ext["id"]] + ext["lag"]
            pass
```
