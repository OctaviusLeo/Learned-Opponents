"""PPO trainer entry point.

Responsibility: single command ``--variant {baseline|randomized} --seed <int>``
that trains the Policy with PPO under the configured ``Training_Budget`` and
writes ``Run_Artifacts`` (resolved config, seed, checkpoint, commit) to
``runs/<variant>/seed<seed>/``. On MJX/RTX 3080, set
``JAX_DEFAULT_MATMUL_PRECISION=highest``. Training budget, PPO hyperparameters,
and seeds are read from ``config/train.yaml``.

Requirements: 4.1, 4.4, 4.5, 4.6, 4.7, 0.10.

[ME] owner implements this core module (tasks.md task 8.1) — Kiro reviews it.
"""

raise NotImplementedError(
    "[ME] owner implements this core module — see tasks.md task 8.1"
)
