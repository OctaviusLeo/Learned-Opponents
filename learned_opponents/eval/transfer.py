"""Transfer check, Python side (task 17.1). Driven by scripts/run_transfer.ps1:

    python -m learned_opponents.eval.transfer episodes   # results/transfer/episodes.json (50 layouts)
    python -m learned_opponents.eval.transfer mujoco     # six policies on those layouts in MuJoCo
    python -m learned_opponents.eval.transfer summarize  # pair with the game results -> summary.md

Both engines play the same 50 layouts (from a fixed seed) at nominal parameters with
deterministic actions, so each variant has 3 seeds x 50 = 150 episodes per engine, and the gap
compares the same tasks.
"""

from __future__ import annotations

import json
import sys
import time

from learned_opponents import config as cfglib
from learned_opponents.env import obstacles as ob
from learned_opponents.eval.robustness import EVAL_DIR, check_runs, provenance, run_dir, write_json
from learned_opponents.randomization.domain_rand import make_rng

TRANSFER = cfglib.REPO_ROOT / "results" / "transfer"
LAYOUT_SEED = 50505
N_LAYOUTS = 50
VARIANTS = ("baseline", "randomized")


def episodes() -> int:
    task = cfglib.load_task()
    lay = ob.sample_layouts(make_rng(LAYOUT_SEED, "transfer_layouts"), N_LAYOUTS, task)
    ob.save_layouts_json(str(TRANSFER / "episodes.json"), lay,
                         {"generated_by": "python -m learned_opponents.eval.transfer episodes",
                          "layout_seed": LAYOUT_SEED, "n_layouts": N_LAYOUTS})
    print(f"wrote results/transfer/episodes.json ({N_LAYOUTS} layouts, seed {LAYOUT_SEED})")
    return 0


def load_layouts(task) -> ob.Layouts:
    data = json.loads((TRANSFER / "episodes.json").read_text(encoding="utf-8"))
    return ob.layouts_from_json(data["episodes"], int(task["n_obstacles"][1]))


def mujoco(allow_dirty: bool) -> int:
    from learned_opponents.eval.rollout import make_eval_env, nominal_params, run_episodes
    from learned_opponents.randomization.domain_rand import parse_ranges
    from learned_opponents.train.policy import load_actor

    commit = provenance(allow_dirty)
    task = cfglib.load_task()
    train = cfglib.load_yaml("train.yaml")
    layouts = load_layouts(task)
    env = make_eval_env(task, len(layouts))
    params = nominal_params(parse_ranges(cfglib.load_yaml("domain_rand.yaml")))
    names = {1: "success", 2: "collision", 3: "timeout"}
    out = {"command": "python -m learned_opponents.eval.transfer mujoco", "commit": commit, "models": {}}
    t0 = time.perf_counter()
    for variant in VARIANTS:
        check_runs(variant, train["seeds"][variant])
        for seed in train["seeds"][variant]:
            actor, _ = load_actor(run_dir(variant, seed), device="cuda:0")
            eps, _ = run_episodes(env, actor, layouts, params, noise_seed=task["eval_seed"])
            out["models"][f"{variant}_seed{seed}"] = [{"index": i, "reason": names[int(r)], "steps": int(s)}
                                                     for i, (r, s) in enumerate(zip(eps.reason, eps.steps))]
    out["wall_s"] = round(time.perf_counter() - t0, 1)
    write_json(TRANSFER / "mujoco_results.json", out)
    print("wrote results/transfer/mujoco_results.json")
    return 0


def summarize(allow_dirty: bool) -> int:
    from learned_opponents.eval import summary

    commit = provenance(allow_dirty)
    train = cfglib.load_yaml("train.yaml")
    mj = json.loads((TRANSFER / "mujoco_results.json").read_text(encoding="utf-8"))
    gm = json.loads((TRANSFER / "game_results.json").read_text(encoding="utf-8"))
    seeds = list(train["seeds"]["baseline"])
    if list(train["seeds"]["randomized"]) != seeds:
        raise SystemExit("transfer pairing assumes the same seed list for both variants")
    data = {"kind": "transfer", "command": "powershell -File scripts/run_transfer.ps1", "commit": commit,
            "mujoco_commit": mj["commit"], "layout_seed": LAYOUT_SEED, "n_layouts": N_LAYOUTS, "seeds": seeds,
            "variants": list(VARIANTS), "game_runtime": gm["runtime"], "engine_version": gm["engine_version"],
            "game_wall_s": gm["wall_s"],
            "mujoco": {k: [e["reason"] for e in v] for k, v in mj["models"].items()},
            "game": {k: [e["reason"] for e in v] for k, v in gm["models"].items()},
            "mujoco_steps": {k: [e["steps"] for e in v] for k, v in mj["models"].items()},
            "game_steps": {k: [e["steps"] for e in v] for k, v in gm["models"].items()}}
    write_json(EVAL_DIR / "transfer.json", data)
    summary.write_summary()
    print("wrote results/eval/transfer.json and results/summary.md")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    allow = "--allow-dirty" in sys.argv
    if cmd == "episodes":
        raise SystemExit(episodes())
    if cmd == "mujoco":
        raise SystemExit(mujoco(allow))
    if cmd == "summarize":
        raise SystemExit(summarize(allow))
    raise SystemExit("usage: python -m learned_opponents.eval.transfer {episodes|mujoco|summarize} [--allow-dirty]")
