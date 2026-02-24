"""
run_experiments.py

Entry point for the Sustainable RCMPSP experiments.

Usage
-----
  python run_experiments.py --mode paper
  python run_experiments.py --mode paper --scenario moderate
  python run_experiments.py --mode paper --pop-size 50 --max-gen 100   # quick test

Modes
-----
  paper  : 1 run per scenario (3 total) with production parameters.
           Outputs per scenario go to  results/paper/{scenario}/:
             pareto_front.csv      Z1 (EUR), Z2 (m3) for each Pareto point
             convergence.csv       gen, best_z1, best_z2  (one row per generation)
             pareto_front.png      scatter Z1 vs Z2; naive baseline marked
             convergence.png       dual-axis Z1* and Z2* vs generation
             solution_analysis.txt human-readable breakdown of 3 key solutions
           A summary table is printed at the end.

  stats  : not implemented yet (defer until reviewers request it).

Default production parameters (--mode paper):
  pop_size=100, T_neighbours=20, max_gen=300, wa_attempts=2, seed=0
  Estimated runtime: ~5 min per scenario (15 min total)

Quick-test parameters:
  pop_size=30, T_neighbours=10, max_gen=50, wa_attempts=1
  Estimated runtime: ~30 s per scenario
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Import project modules ────────────────────────────────────────────────────

# Allow running as  python run_experiments.py  from the repo root
sys.path.insert(0, str(Path(__file__).parent))

from src.loader import Problem, load
from src.sgs import decode
from src.evaluator import Objectives, evaluate, project_breakdown
from src.scenario import generate_epsilon, SCENARIO_NAMES
from src.moead import MOEADResult, run_moead


# ── Matplotlib (optional) ─────────────────────────────────────────────────────

try:
    import matplotlib
    matplotlib.use("Agg")           # non-interactive backend (no display needed)
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sustainable RCMPSP experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode", required=True, choices=["paper", "stats"],
                   help="Experiment mode")
    p.add_argument("--scenario", default=None,
                   choices=list(SCENARIO_NAMES) + ["all"],
                   help="Which scenario to run (default: all three)")
    p.add_argument("--pop-size",    type=int, default=100, metavar="N")
    p.add_argument("--max-gen",     type=int, default=300, metavar="G")
    p.add_argument("--T",           type=int, default=20,  metavar="T",
                   help="Neighbourhood size for MOEA/D")
    p.add_argument("--wa-attempts", type=int, default=2,   metavar="A",
                   help="Water-arbitrage attempts per offspring")
    p.add_argument("--seed",        type=int, default=0,
                   help="RNG seed (paper mode uses one seed)")
    p.add_argument("--out-dir", default="results", metavar="DIR",
                   help="Root output directory (default: results/)")
    return p.parse_args()


# ── Naive baseline ────────────────────────────────────────────────────────────

def compute_naive_baseline(prob: Problem, epsilon: List[float]) -> Objectives:
    """
    Evaluate the identity-priority schedule (FIFO by global ID).

    This is the 'no water awareness' reference: activities are scheduled
    strictly in global-ID order without any attempt to avoid high-eps days.
    """
    priority = list(range(len(prob.activities)))
    start = decode(priority, prob)
    return evaluate(start, prob, epsilon)


# ── Representative solution selection ────────────────────────────────────────

def _pick_representatives(
    pareto_front: List[Objectives],
) -> Dict[str, int]:
    """
    Select three representative indices from the Pareto front:
      - 'min_z1'    : minimum tardiness (best on Z1)
      - 'min_z2'    : minimum water violations (best on Z2)
      - 'balanced'  : closest to normalised utopia (0, 0)
    Returns a dict {label: index}.
    """
    n = len(pareto_front)
    if n == 1:
        return {"min_z1": 0, "min_z2": 0, "balanced": 0}

    z1s = [o.z1 for o in pareto_front]
    z2s = [o.z2 for o in pareto_front]
    z1_min, z1_max = min(z1s), max(z1s)
    z2_min, z2_max = min(z2s), max(z2s)

    idx_min_z1 = z1s.index(z1_min)
    idx_min_z2 = z2s.index(z2_min)

    # Normalised distance to utopia (0, 0)
    scale1 = (z1_max - z1_min) or 1.0
    scale2 = (z2_max - z2_min) or 1.0
    dists = [
        ((o.z1 - z1_min) / scale1) ** 2 + ((o.z2 - z2_min) / scale2) ** 2
        for o in pareto_front
    ]
    # Balanced point: minimise sum of normalised objectives (not corner points)
    balanced_candidates = [
        i for i in range(n)
        if i not in (idx_min_z1, idx_min_z2)
    ]
    if balanced_candidates:
        idx_balanced = min(balanced_candidates, key=lambda i: dists[i])
    else:
        idx_balanced = idx_min_z1  # fallback: only 1-2 points

    return {"min_z1": idx_min_z1, "min_z2": idx_min_z2, "balanced": idx_balanced}


# ── Output writers ────────────────────────────────────────────────────────────

def _save_pareto_csv(pareto_front: List[Objectives], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "z1_eur", "z2_m3"])
        for i, obj in enumerate(pareto_front):
            w.writerow([i + 1, f"{obj.z1:.2f}", f"{obj.z2:.2f}"])


def _save_convergence_csv(
    history: List[Tuple[int, float, float]], path: Path
) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["generation", "best_z1_eur", "best_z2_m3"])
        for gen, z1, z2 in history:
            w.writerow([gen, f"{z1:.2f}", f"{z2:.2f}"])


def _save_convergence_plot(
    history: List[Tuple[int, float, float]], scenario: str, path: Path
) -> None:
    gens  = [h[0] for h in history]
    z1s   = [h[1] for h in history]
    z2s   = [h[2] for h in history]

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()

    ax1.plot(gens, z1s, color="tab:blue",  label="Z1* (EUR)")
    ax2.plot(gens, z2s, color="tab:orange", label="Z2* (m3)", linestyle="--")

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Best Z1 (EUR)", color="tab:blue")
    ax2.set_ylabel("Best Z2 (m3)", color="tab:orange")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax1.set_title(f"Convergence — scenario {scenario}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_pareto_plot(
    pareto_front: List[Objectives],
    naive: Objectives,
    scenario: str,
    reps: Dict[str, int],
    path: Path,
) -> None:
    z1s = [o.z1 for o in pareto_front]
    z2s = [o.z2 for o in pareto_front]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(z1s, z2s, c="steelblue", s=40, zorder=3, label="Pareto front")
    ax.plot(z1s, z2s, c="steelblue", linewidth=0.8, zorder=2)

    # Mark representative solutions
    colors = {"min_z1": "green", "min_z2": "red", "balanced": "purple"}
    labels_map = {
        "min_z1": "Min Z1 (min delay)",
        "min_z2": "Min Z2 (min water viol.)",
        "balanced": "Balanced",
    }
    plotted = set()
    for label, idx in reps.items():
        if idx in plotted:
            continue
        plotted.add(idx)
        obj = pareto_front[idx]
        ax.scatter(obj.z1, obj.z2, c=colors[label], s=100, zorder=5,
                   label=labels_map[label], edgecolors="black", linewidths=0.5)

    # Naive baseline
    ax.scatter(naive.z1, naive.z2, marker="*", c="black", s=200, zorder=5,
               label=f"Naive baseline")

    ax.set_xlabel("Z1 — Tardiness penalty (EUR)")
    ax.set_ylabel("Z2 — Water quota violation (m3)")
    ax.set_title(f"Pareto front — scenario {scenario}  (|PF|={len(pareto_front)})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_solution_analysis(
    pareto_front: List[Objectives],
    pareto_solutions: List[List[int]],
    reps: Dict[str, int],
    naive_obj: Objectives,
    prob: Problem,
    epsilon: List[float],
    scenario: str,
    path: Path,
) -> None:
    lines: List[str] = []

    def w(s: str = "") -> None:
        lines.append(s)

    w(f"Solution Analysis — scenario: {scenario}")
    w("=" * 60)
    w()

    # --- Naive baseline ---
    w("NAIVE BASELINE (identity priority, no water awareness)")
    w("-" * 60)
    w(f"  Z1 = {naive_obj.z1:>12,.0f} EUR")
    w(f"  Z2 = {naive_obj.z2:>12,.1f} m3")
    naive_bd = project_breakdown(
        decode(list(range(len(prob.activities))), prob), prob, epsilon
    )
    for pname, d in naive_bd.items():
        w(f"  {pname}: tardiness={d['tardiness_days']:4d}d  "
          f"penalty={d['penalty_eur']:>9,.0f} EUR  "
          f"water_violation={d['water_violation_m3']:>8,.0f} m3")
    w()

    # --- Pareto front summary ---
    w(f"PARETO FRONT SUMMARY  (|PF| = {len(pareto_front)})")
    w("-" * 60)
    z1s = [o.z1 for o in pareto_front]
    z2s = [o.z2 for o in pareto_front]
    w(f"  Z1 range: {min(z1s):>10,.0f} — {max(z1s):>10,.0f} EUR")
    w(f"  Z2 range: {min(z2s):>10,.1f} — {max(z2s):>10,.1f} m3")
    w(f"  vs naive baseline:")
    if naive_obj.z1 > 0:
        best_z1_save = (naive_obj.z1 - min(z1s)) / naive_obj.z1 * 100
        w(f"    Best Z1 reduction : {best_z1_save:+.1f}%")
    if naive_obj.z2 > 0:
        best_z2_save = (naive_obj.z2 - min(z2s)) / naive_obj.z2 * 100
        w(f"    Best Z2 reduction : {best_z2_save:+.1f}%")
    w()

    # --- Three representative solutions ---
    seen_idx: set = set()
    label_order = ["min_z1", "balanced", "min_z2"]
    label_titles = {
        "min_z1":    "SOLUTION A — Minimum tardiness  (best Z1)",
        "min_z2":    "SOLUTION C — Minimum water violations  (best Z2)",
        "balanced":  "SOLUTION B — Balanced (closest to normalised utopia)",
    }

    for label in label_order:
        idx = reps[label]
        if idx in seen_idx:
            continue
        seen_idx.add(idx)
        obj = pareto_front[idx]
        sol = pareto_solutions[idx]

        w(label_titles[label])
        w("-" * 60)
        w(f"  Z1 = {obj.z1:>12,.0f} EUR   "
          f"({'same' if obj.z1 == naive_obj.z1 else f'{(obj.z1-naive_obj.z1)/max(naive_obj.z1,1)*100:+.1f}%'} vs naive)")
        w(f"  Z2 = {obj.z2:>12,.1f} m3    "
          f"({'same' if obj.z2 == naive_obj.z2 else f'{(obj.z2-naive_obj.z2)/max(naive_obj.z2,1)*100:+.1f}%'} vs naive)")

        bd = project_breakdown(decode(sol, prob), prob, epsilon)
        for pname, d in bd.items():
            w(f"  {pname}: tardiness={d['tardiness_days']:4d}d  "
              f"penalty={d['penalty_eur']:>9,.0f} EUR  "
              f"constr_days={d['n_construction_days']:4d}  "
              f"eps_sum={d['eps_sum']:6.1f}  "
              f"water_violation={d['water_violation_m3']:>8,.0f} m3")
        w()

    path.write_text("\n".join(lines), encoding="utf-8")


# ── Paper mode ────────────────────────────────────────────────────────────────

def run_paper_mode(args: argparse.Namespace) -> None:
    print("=" * 65)
    print("  Sustainable RCMPSP — Paper mode")
    print(f"  pop_size={args.pop_size}  max_gen={args.max_gen}  "
          f"T={args.T}  wa={args.wa_attempts}  seed={args.seed}")
    if not HAS_MPL:
        print("  [WARNING] matplotlib not found — skipping plots")
    print("=" * 65)

    # Which scenarios to run
    scenarios: List[str] = (
        list(SCENARIO_NAMES)
        if args.scenario in (None, "all")
        else [args.scenario]
    )

    print(f"\nLoading portfolio... ", end="", flush=True)
    prob = load()
    print(f"OK ({len(prob.activities)} activities, horizon={prob.horizon} days)")

    out_root = Path(args.out_dir) / "paper"
    out_root.mkdir(parents=True, exist_ok=True)

    # Summary table rows collected across scenarios
    summary_rows: List[dict] = []

    total_t0 = time.time()

    for scenario in scenarios:
        print(f"\n{'-'*65}")
        print(f"  Scenario: {scenario}")
        print(f"{'-'*65}")

        sc_dir = out_root / scenario
        sc_dir.mkdir(parents=True, exist_ok=True)

        # Generate epsilon series
        print(f"  Generating epsilon series (seed={args.seed})... ", end="", flush=True)
        epsilon = generate_epsilon(scenario, prob.horizon, seed=args.seed)
        n_restrict = sum(1 for e in epsilon if e > 0)
        print(f"OK  restrict_days={n_restrict}/{prob.horizon} "
              f"({100*n_restrict/prob.horizon:.0f}%)  "
              f"mean_eps={sum(epsilon)/len(epsilon):.3f}")

        # Naive baseline
        print(f"  Computing naive baseline... ", end="", flush=True)
        naive_obj = compute_naive_baseline(prob, epsilon)
        print(f"Z1={naive_obj.z1:,.0f} EUR  Z2={naive_obj.z2:,.1f} m3")

        # Run MOEA/D
        print(f"  Running MOEA/D...", flush=True)
        result: MOEADResult = run_moead(
            prob,
            epsilon,
            pop_size=args.pop_size,
            T_neighbours=args.T,
            max_gen=args.max_gen,
            wa_attempts=args.wa_attempts,
            seed=args.seed,
            verbose=True,
            log_every=max(1, args.max_gen // 10),
        )

        pf   = result.pareto_front
        psol = result.pareto_solutions

        # Representative solutions
        reps = _pick_representatives(pf)

        # Save outputs
        print(f"  Saving outputs to {sc_dir}/")

        _save_pareto_csv(pf, sc_dir / "pareto_front.csv")
        _save_convergence_csv(result.history, sc_dir / "convergence.csv")
        _save_solution_analysis(
            pf, psol, reps, naive_obj, prob, epsilon, scenario,
            sc_dir / "solution_analysis.txt"
        )

        if HAS_MPL:
            _save_convergence_plot(result.history, scenario,
                                   sc_dir / "convergence.png")
            _save_pareto_plot(pf, naive_obj, scenario, reps,
                              sc_dir / "pareto_front.png")
            print(f"    convergence.png, pareto_front.png saved")

        print(f"    pareto_front.csv, convergence.csv, solution_analysis.txt saved")

        # Collect summary row
        z1s = [o.z1 for o in pf]
        z2s = [o.z2 for o in pf]
        summary_rows.append({
            "scenario":      scenario,
            "pf_size":       len(pf),
            "z1_min":        min(z1s),
            "z1_max":        max(z1s),
            "z2_min":        min(z2s),
            "z2_max":        max(z2s),
            "naive_z1":      naive_obj.z1,
            "naive_z2":      naive_obj.z2,
            "elapsed_sec":   result.elapsed_sec,
        })

    # Write summary CSV
    summary_csv = out_root / "summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["scenario", "pf_size",
                      "z1_min", "z1_max", "z2_min", "z2_max",
                      "naive_z1", "naive_z2", "elapsed_sec"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in summary_rows:
            w.writerow({k: (f"{v:.2f}" if isinstance(v, float) else v)
                        for k, v in row.items()})

    total_elapsed = time.time() - total_t0

    # Print summary table
    print(f"\n{'='*65}")
    print("  SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Scenario':<10} {'|PF|':>5}  "
          f"{'Z1 min':>12}  {'Z1 max':>12}  "
          f"{'Z2 min':>12}  {'Z2 max':>12}  "
          f"{'Time':>8}")
    print(f"  {'-'*10:<10} {'-'*5:>5}  "
          f"{'-'*12:>12}  {'-'*12:>12}  "
          f"{'-'*12:>12}  {'-'*12:>12}  "
          f"{'-'*8:>8}")
    for row in summary_rows:
        print(f"  {row['scenario']:<10} {row['pf_size']:>5}  "
              f"{row['z1_min']:>12,.0f}  {row['z1_max']:>12,.0f}  "
              f"{row['z2_min']:>12,.1f}  {row['z2_max']:>12,.1f}  "
              f"{row['elapsed_sec']:>7.1f}s")
    print()
    print(f"  Total elapsed: {total_elapsed:.1f}s")
    print(f"  Outputs written to: {out_root}/")
    print(f"  Summary table: {summary_csv}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    if args.mode == "paper":
        run_paper_mode(args)
    elif args.mode == "stats":
        print("ERROR: --mode stats is not yet implemented.")
        print("Run --mode paper first; stats mode will be added if reviewers request it.")
        sys.exit(1)


if __name__ == "__main__":
    main()
