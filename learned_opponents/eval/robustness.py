"""Robustness grid evaluation.

Responsibility: sweep the ``Robustness_Grid`` (mass, wheel friction, latency —
each including >= 1 value outside the training range) and report per cell the
Success_Rate, Collision_Rate, and mean Time_To_Goal, each with a 95% CI over
``episodes_per_cell``, aggregated across the >= 3 seeds with the aggregation
method stated. Also provides the nominal-evaluation entry point.

Requirements: 6.1, 6.2, 6.3 (and 5.1-5.3 for the nominal eval entry point).

[ME] owner implements this core evaluation module (tasks.md task 11.1) —
Kiro reviews it.
"""

raise NotImplementedError(
    "[ME] owner implements this core module — see tasks.md task 11.1"
)
