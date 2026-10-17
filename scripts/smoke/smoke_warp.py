"""M0 smoke test for mjlab / MuJoCo Warp, task 2.1.

Run with the lo-mjlab environment. Steps the shared smoke arena through
MuJoCo Warp (the simulator mjlab drives) with all worlds batched on the GPU,
reads the rangefinder block every step, and checks a PyTorch -> ONNX export of
an rsl-rl style actor MLP (the policy type mjlab trains) against ONNX Runtime.
Writes a JSON dict of per-condition results to --out.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from importlib import metadata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arena  # noqa: E402


def versions() -> dict:
    out = {}
    for pkg in ["mjlab", "mujoco", "mujoco-warp", "warp-lang", "torch", "rsl-rl-lib",
                "onnx", "onnxruntime", "onnxscript", "numpy"]:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return out


def run(num_envs: int, n_steps: int) -> dict:
    res: dict = {"framework": "mjlab", "implementation": "MuJoCo Warp (mujoco_warp, as driven by mjlab)",
                 "num_envs": num_envs, "n_steps": n_steps, "conditions": {}, "versions": versions()}
    cond = res["conditions"]

    import mujoco
    import mujoco_warp as mjw
    import numpy as np
    import torch
    import warp as wp

    wp.init()
    dev = wp.get_device()
    res["warp_device"] = str(dev)
    res["torch_cuda"] = torch.cuda.is_available()
    res["torch_device_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    cond["gpu_backend"] = bool(dev.is_cuda and torch.cuda.is_available())

    mjm = mujoco.MjModel.from_xml_string(arena.build_xml())
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    n_rays = arena.N_RAYS
    m = mjw.put_model(mjm)
    # nconmax / njmax are per world: 4 floor contacts, at most 10 constraint rows.
    d = mjw.put_data(mjm, mjd, nworld=num_envs, nconmax=8, njmax=32)

    # Per-world start pose (x, y, yaw) and random wheel commands so rays differ.
    g = torch.Generator(device="cuda").manual_seed(0)
    qpos = wp.to_torch(d.qpos)
    ctrl = wp.to_torch(d.ctrl)
    xy = (torch.rand((num_envs, 2), generator=g, device="cuda") - 0.5)
    yaw = (torch.rand((num_envs,), generator=g, device="cuda") * 2 - 1) * np.pi
    qpos[:, 0:2] = xy
    qpos[:, 3] = torch.cos(yaw / 2)
    qpos[:, 4:6] = 0.0
    qpos[:, 6] = torch.sin(yaw / 2)
    ctrl[:] = (torch.rand((num_envs, 2), generator=g, device="cuda") * 2 - 1) * arena.MAX_WHEEL_SPEED
    mjw.forward(m, d)

    sens_t = wp.to_torch(d.sensordata)
    res["sensordata_shape_per_step"] = list(sens_t.shape)

    # Warm-up + graph capture (mjlab also steps through a captured CUDA graph).
    mjw.step(m, d)
    wp.synchronize()
    with wp.ScopedCapture() as cap:
        mjw.step(m, d)
    graph = cap.graph

    # Correctness pass: read the rangefinder block every step.
    rays_all = torch.empty((n_steps, num_envs, n_rays), device="cuda")
    up_all = torch.empty((n_steps, num_envs), device="cuda")
    for t in range(n_steps):
        wp.capture_launch(graph)
        rays_all[t] = sens_t[:, :n_rays]
        up_all[t] = sens_t[:, n_rays]
    wp.synchronize()
    rays = rays_all.cpu().numpy()
    up = up_all.cpu().numpy()
    cond["steps_on_gpu_at_count"] = bool(cond["gpu_backend"] and sens_t.shape[0] == num_envs)
    cond["range_rays_every_step"] = bool(rays.shape == (n_steps, num_envs, n_rays) and np.isfinite(rays).all())
    hits = rays[rays >= 0]
    res["ray_hit_range_m"] = [float(hits.min()), float(hits.max())] if hits.size else None
    res["ray_hit_fraction"] = float((rays >= 0).mean())
    res["miss_value_observed"] = sorted({float(v) for v in np.unique(up)})
    cond["miss_value_proven"] = bool(np.all(up == up.flat[0]) and up.flat[0] < 0)
    cond["no_self_hits"] = bool(hits.size == 0 or hits.min() > 0.15)
    q = wp.to_torch(d.qpos).cpu().numpy()
    res["robot_z_range_m"] = [float(q[:, 2].min()), float(q[:, 2].max())]
    cond["physics_finite"] = bool(np.isfinite(q).all())

    # Throughput: the same number of steps, timed after warm-up.
    reps = 3
    wp.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps * n_steps):
        wp.capture_launch(graph)
    wp.synchronize()
    dt = time.perf_counter() - t0
    res["wall_s"] = dt
    res["env_steps_per_s"] = num_envs * n_steps * reps / dt

    try:
        res["onnx"] = onnx_check(n_rays + 5)
        cond["onnx_export_path"] = res["onnx"]["max_abs_diff"] <= 1e-4
    except Exception as e:  # recorded, not fatal: tie-break only
        res["onnx"] = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]}
        cond["onnx_export_path"] = False
    return res


def onnx_check(obs_dim: int) -> dict:
    import io

    import numpy as np
    import onnxruntime as ort
    import torch
    from torch import nn

    torch.manual_seed(1)

    class Actor(nn.Module):
        """rsl-rl actor layout: obs normalizer -> MLP(ELU) -> mean; deterministic = mean."""

        def __init__(self):
            super().__init__()
            self.register_buffer("mean", torch.linspace(-0.1, 0.1, obs_dim))
            self.register_buffer("std", torch.linspace(0.5, 1.5, obs_dim))
            self.mlp = nn.Sequential(nn.Linear(obs_dim, 32), nn.ELU(), nn.Linear(32, 32), nn.ELU(), nn.Linear(32, 2))

        def forward(self, obs):
            return torch.clamp(self.mlp((obs - self.mean) / self.std), -1.0, 1.0)

    actor = Actor().eval()
    obs = torch.rand((64, obs_dim)) * 2 - 1
    with torch.no_grad():
        ref = actor(obs).numpy()
    buf = io.BytesIO()
    torch.onnx.export(actor, (obs[:1],), buf, input_names=["obs"], output_names=["action"],
                      dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}}, opset_version=17, dynamo=False)
    sess = ort.InferenceSession(buf.getvalue(), providers=["CPUExecutionProvider"])
    out = sess.run(["action"], {"obs": obs.numpy().astype(np.float32)})[0]
    return {"exporter": "torch.onnx.export (TorchScript exporter)", "opset": 17,
            "n_obs": int(obs.shape[0]), "max_abs_diff": float(np.abs(out - ref).max())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    try:
        res = run(args.num_envs, args.steps)
    except Exception as e:
        res = {"framework": "mjlab", "error": f"{type(e).__name__}: {e}",
               "trace": traceback.format_exc()[-4000:], "conditions": {}, "versions": versions()}
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "trace"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
