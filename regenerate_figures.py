"""
regenerate_figures.py

Reads the existing CSVs in results/paper/{scenario}/ and regenerates
all figures at high DPI — no need to re-run the experiments.

Output: paper/ECAM_submission_with_authors/figures/

Usage:
    python regenerate_figures.py
    python regenerate_figures.py --dpi 1200   # default
    python regenerate_figures.py --dpi 300    # quick check
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Config ────────────────────────────────────────────────────────────────────

RESULTS_ROOT = Path("results/paper")

SUBMISSION_DIRS = [
    Path("paper/ECAM_submission_with_authors/figures"),
]

SCENARIOS = ["mild", "moderate", "severe"]

# Naive baseline values read from solution_analysis.txt (no need to recompute)
NAIVE = {
    "mild":     {"z1": 36_500.0,  "z2": 44_609.5},
    "moderate": {"z1": 36_500.0,  "z2": 98_472.5},
    "severe":   {"z1": 36_500.0,  "z2": 174_981.7},
}

# Output filenames expected by the submission (per figures/README.txt)
PARETO_NAME     = "pareto_front_{scenario}.png"
CONVERGENCE_NAME = "convergence_{scenario}.png"   # only mild needed, but gen all


# ── Readers ───────────────────────────────────────────────────────────────────

def read_pareto_csv(path: Path):
    """Returns list of (z1, z2) tuples."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append((float(row["z1_eur"]), float(row["z2_m3"])))
    return rows


def read_convergence_csv(path: Path):
    """Returns list of (generation, z1, z2) tuples."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append((int(row["generation"]), float(row["best_z1_eur"]), float(row["best_z2_m3"])))
    return rows


# ── Plotters ──────────────────────────────────────────────────────────────────

def plot_pareto(pareto, naive_z1, naive_z2, scenario, out_path, dpi):
    z1s = [p[0] for p in pareto]
    z2s = [p[1] for p in pareto]

    # Representative points (same logic as run_experiments.py)
    z1_min, z1_max = min(z1s), max(z1s)
    z2_min, z2_max = min(z2s), max(z2s)
    scale1 = (z1_max - z1_min) or 1.0
    scale2 = (z2_max - z2_min) or 1.0

    idx_min_z1 = z1s.index(z1_min)
    idx_min_z2 = z2s.index(z2_min)

    n = len(pareto)
    dists = [
        ((z1s[i] - z1_min) / scale1) ** 2 + ((z2s[i] - z2_min) / scale2) ** 2
        for i in range(n)
    ]
    candidates = [i for i in range(n) if i not in (idx_min_z1, idx_min_z2)]
    idx_balanced = min(candidates, key=lambda i: dists[i]) if candidates else idx_min_z1

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(z1s, z2s, c="steelblue", s=40, zorder=3, label="Pareto front")
    ax.plot(z1s, z2s, c="steelblue", linewidth=0.8, zorder=2)

    rep_cfg = [
        (idx_min_z1,  "green",  "Min Z1 (min delay)"),
        (idx_min_z2,  "red",    "Min Z2 (min water viol.)"),
        (idx_balanced,"purple", "Balanced"),
    ]
    plotted = set()
    for idx, color, label in rep_cfg:
        if idx in plotted:
            continue
        plotted.add(idx)
        ax.scatter(z1s[idx], z2s[idx], c=color, s=100, zorder=5,
                   label=label, edgecolors="black", linewidths=0.5)

    ax.scatter(naive_z1, naive_z2, marker="*", c="black", s=200,
               zorder=5, label="Naive baseline")

    ax.set_xlabel("Z1 — Tardiness penalty (EUR)")
    ax.set_ylabel("Z2 — Water quota violation (m³)")
    ax.set_title(f"Pareto front — scenario {scenario}  (|PF|={len(pareto)})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(history, scenario, out_path, dpi):
    gens = [h[0] for h in history]
    z1s  = [h[1] for h in history]
    z2s  = [h[2] for h in history]

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()

    ax1.plot(gens, z1s, color="tab:blue",   label="Z1* (EUR)")
    ax2.plot(gens, z2s, color="tab:orange", label="Z2* (m³)", linestyle="--")

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Best Z1 (EUR)",  color="tab:blue")
    ax2.set_ylabel("Best Z2 (m³)",   color="tab:orange")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax1.set_title(f"Convergence — scenario {scenario}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Regenerate figures from CSVs at high DPI")
    parser.add_argument("--dpi", type=int, default=1200,
                        help="Output DPI (default: 1200)")
    args = parser.parse_args()

    # Ensure output dirs exist
    for d in SUBMISSION_DIRS:
        d.mkdir(parents=True, exist_ok=True)

    for scenario in SCENARIOS:
        sc_dir = RESULTS_ROOT / scenario
        pareto_csv = sc_dir / "pareto_front.csv"
        conv_csv   = sc_dir / "convergence.csv"

        if not pareto_csv.exists() or not conv_csv.exists():
            print(f"[SKIP] {scenario}: CSVs not found in {sc_dir}")
            continue

        print(f"[{scenario}] Reading data...", end=" ", flush=True)
        pareto  = read_pareto_csv(pareto_csv)
        history = read_convergence_csv(conv_csv)
        naive   = NAIVE[scenario]
        print(f"Pareto points: {len(pareto)}, generations: {len(history)}")

        for out_dir in SUBMISSION_DIRS:
            pf_path   = out_dir / PARETO_NAME.format(scenario=scenario)
            conv_path = out_dir / CONVERGENCE_NAME.format(scenario=scenario)

            plot_pareto(pareto, naive["z1"], naive["z2"], scenario, pf_path, args.dpi)
            plot_convergence(history, scenario, conv_path, args.dpi)

            print(f"  -> {pf_path}")
            print(f"  -> {conv_path}")

    print(f"\nDone. All figures saved at {args.dpi} dpi.")


if __name__ == "__main__":
    main()
