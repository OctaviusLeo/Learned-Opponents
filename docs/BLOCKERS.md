# Blockers and environment failures

## 2026-10-17 22:16 EDT: robustness grid job vanished (task 11.1)

- Command: `bash scripts/run_detached.sh robustness_grid ~/.venvs/learned-opponents/bin/python -m learned_opponents.eval.robustness` at commit `5e8aa94`.
- Symptom: after 28 minutes `runs/logs/robustness_grid.log` was still empty (0 bytes: not even the import-time warnings every run prints), the wrapper process (pid 131149) and its Python child were gone, and no `runs/logs/robustness_grid.exit` was written. `dmesg` shows no out-of-memory kill visible to this user; `MemAvailable` was 11.3 GB of 16.3 GB.
- Context: jobs from four other repositories were running in the same WSL distro and on the same GPU at the time (Grounded-QA-Service uvicorn, Finetune-And-Serve `scripts/gpu_job.sh python -m fts.baselines`, Autonomy-Test-Harness Docker + ROS 2 simulation, Agent-Injection-Bench tests), with 7.1 GB of GPU memory in use by them. An earlier detached evaluation in this project (pilot curve, 21:05) also stalled with the GPU at 10 GB and was restarted by hand.
- Suspected cause: interference from the concurrent workloads (a process-group kill or GPU memory exhaustion); not confirmed.
- Action: the sweep (expected under 30 minutes) is re-run once in the foreground instead of detached; the result is recorded below. No target, test, seed, or episode count is changed.
- Result: the foreground re-run (`runs/logs/robustness_grid_fg.log`) completed all 102 evaluations in 435 s. It was started with an unnecessary `--allow-dirty` flag (only this file was uncommitted, which the clean-tree rule already allows), so its output was discarded and the sweep was run once more in the foreground from the commit that adds this note; that run is the one reported in `results/summary.md`. Status: resolved by running in the foreground; the underlying cause of the vanished detached job remains unknown.
