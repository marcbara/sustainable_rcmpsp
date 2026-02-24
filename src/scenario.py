"""
scenario.py

Generate ε_t (daily water scarcity shock) series for the three scenarios.

Model: Q_t = Q̄ × (1 − ε_t)
  ε_t = 0   → full quota available (no restriction)
  ε_t = 0.5 → quota cut by 50 %

We use Sample Average Approximation (SAA): generate the ε_t series ONCE
before running the optimiser, then treat it as deterministic input. The
randomness lives in the experimental design (different seeds = different
realisations), not inside the algorithm.

Distribution model (mixture: Bernoulli × severity):
  Each day independently:
    with probability  p_restrict → ε_t ~ Uniform(sev_low, sev_high)
    with probability  1−p_restrict → ε_t = 0  (no restriction)

  This creates clear "restriction days" and "free days," giving the
  water-arbitrage operator real temporal variation to exploit — i.e., the
  scheduler can meaningfully avoid restriction days.

  Scenario   | p_restrict | sev_low | sev_high | E[ε_t]
  -----------|------------|---------|----------|-------
  mild       |   0.20     |   0.40  |   0.60   |  0.10
  moderate   |   0.40     |   0.55  |   0.75   |  0.26
  severe     |   0.65     |   0.65  |   0.90   |  0.50

  E[ε_t] = p_restrict × (sev_low + sev_high) / 2

NOTE: Marisa should confirm the exact parameters once she has a bibliographic
reference for water restriction distributions in construction projects.
The interface is easy to update: change _PARAMS values only.
"""

from __future__ import annotations

import numpy as np
from typing import List

# ── Scenario definitions ───────────────────────────────────────────────────────

SCENARIO_NAMES = ("mild", "moderate", "severe")

_PARAMS: dict = {
    # Markov chain with two states: Restriction (R) and Free (F).
    # avg_restrict_days: expected length of a drought spell (days)
    # avg_free_days:     expected length of a free period (days)
    # → p_rr = 1 - 1/avg_restrict_days  (stay-in-restriction probability)
    # → p_ff = 1 - 1/avg_free_days       (stay-in-free probability)
    # Stationary restriction fraction ≈ avg_restrict/(avg_restrict + avg_free)
    #
    # sev_low/sev_high: ε_t on restriction days ~ Uniform(sev_low, sev_high)
    #
    # Physical justification: droughts last 1-3 weeks; free periods 2-5 weeks.
    # This gives the water-arbitrage real temporal windows to exploit.
    #
    # Scenario     | spell(R) | spell(F) | frac_R | E[ε_t]
    # -------------|----------|----------|--------|-------
    # mild         |  7 days  | 28 days  |  20 %  |  0.10
    # moderate     | 14 days  | 21 days  |  40 %  |  0.26
    # severe       | 21 days  | 14 days  |  60 %  |  0.47
    #
    # NOTE: confirm parameters with Marisa and add a bibliographic reference
    # for water restriction distributions in Mediterranean construction.
    "mild":     {"avg_restrict": 7,  "avg_free": 28, "sev_low": 0.40, "sev_high": 0.60},
    "moderate": {"avg_restrict": 14, "avg_free": 21, "sev_low": 0.55, "sev_high": 0.75},
    "severe":   {"avg_restrict": 21, "avg_free": 14, "sev_low": 0.65, "sev_high": 0.90},
}


# ── Generator ─────────────────────────────────────────────────────────────────

def generate_epsilon(scenario: str, horizon: int, seed: int = 0) -> List[float]:
    """
    Generate a deterministic ε_t series of length `horizon`.

    Uses a two-state Markov chain to create drought spells with persistence:
    - State R (restriction): ε_t ~ Uniform(sev_low, sev_high)
    - State F (free): ε_t = 0

    Parameters
    ----------
    scenario : str
        One of "mild", "moderate", "severe".
    horizon : int
        Number of days (= Problem.horizon).
    seed : int
        RNG seed for reproducibility.  Different seeds → different SAA
        realisations of the same scenario.

    Returns
    -------
    List[float]
        epsilon[t] for t = 0 … horizon-1.  Values in [0, 1).
    """
    if scenario not in _PARAMS:
        raise ValueError(f"Unknown scenario '{scenario}'. "
                         f"Choose from: {list(_PARAMS.keys())}")
    p = _PARAMS[scenario]
    rng = np.random.default_rng(seed)

    # Transition probabilities
    p_rr = 1.0 - 1.0 / p["avg_restrict"]   # stay in R
    p_ff = 1.0 - 1.0 / p["avg_free"]        # stay in F

    # Stationary distribution: start in R with prob p_restrict
    p_restrict_stationary = (1 - p_ff) / ((1 - p_rr) + (1 - p_ff))
    state_r = rng.random() < p_restrict_stationary   # True = restriction

    epsilon = np.zeros(horizon)
    severity_draws = rng.uniform(p["sev_low"], p["sev_high"], size=horizon)
    transition_draws = rng.random(size=horizon)

    for t in range(horizon):
        if state_r:
            epsilon[t] = severity_draws[t]
            # Stay in R or transition to F
            state_r = transition_draws[t] < p_rr
        else:
            # Stay in F or transition to R
            state_r = transition_draws[t] >= p_ff

    return epsilon.tolist()


def describe(epsilon: List[float], scenario: str, seed: int) -> str:
    """One-line summary of an ε_t series."""
    arr = np.array(epsilon)
    n_days = len(arr)
    n_restrict = int((arr > 0).sum())
    return (f"scenario={scenario} seed={seed} "
            f"restrict_days={n_restrict}/{n_days} ({100*n_restrict/n_days:.0f}%) "
            f"mean_eps={arr.mean():.3f} mean_on_restrict={arr[arr>0].mean():.3f} "
            f"max={arr.max():.3f}")


# ── Quick self-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    horizon = 1500  # typical for this portfolio

    for sc in SCENARIO_NAMES:
        eps = generate_epsilon(sc, horizon, seed=0)
        print(describe(eps, sc, seed=0))
