"""Navigation environment for the differential-drive robot.

Responsibility: ``reset``/``step``/reward/termination for a differential-drive
robot in an arena with randomized obstacles. ``reset`` places robot, goal, and
obstacles and returns the initial Observation; ``step`` applies wheel-velocity
commands and returns next Observation / reward / termination; success within
``arrival_tolerance_m``, collision on contact, timeout at ``episode_limit``;
reward is the configured weighted sum of progress / success / collision terms.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.7.

[ME] owner implements this core module (tasks.md task 5.1) — Kiro reviews it.
"""

raise NotImplementedError(
    "[ME] owner implements this core module — see tasks.md task 5.1"
)
