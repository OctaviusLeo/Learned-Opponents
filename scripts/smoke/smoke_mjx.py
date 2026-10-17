"""M0 smoke test for MuJoCo Playground / MJX (JAX implementation), task 2.1.

Run with the lo-mjx environment and JAX_DEFAULT_MATMUL_PRECISION=highest.
Writes a JSON dict of per-condition results to the path given by --out.
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
    for pkg in ["playground", "mujoco", "mujoco-mjx", "jax", "jaxlib", "jax-cuda12-plugin",
                "jax-cuda12-pjrt", "brax", "flax", "onnx", "onnxruntime", "numpy"]:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return out


def run(num_envs: int, n_steps: int) -> dict:
    res: dict = {"framework": "mjx", "implementation": "MJX (JAX)", "num_envs": num_envs,
                 "n_steps": n_steps, "conditions": {}, "versions": versions(),
                 "env": {k: os.environ.get(k) for k in ["JAX_DEFAULT_MATMUL_PRECISION",
                                                        "XLA_PYTHON_CLIENT_PREALLOCATE"]}}
    cond = res["conditions"]

    import jax
    import jax.numpy as jp
    import mujoco
    import numpy as np
    from mujoco import mjx

    res["jax_devices"] = [str(d) for d in jax.devices()]
    res["jax_backend"] = jax.default_backend()
    cond["gpu_backend"] = jax.default_backend() == "gpu"

    m = mujoco.MjModel.from_xml_string(arena.build_xml())
    n_rays = arena.N_RAYS
    mx = mjx.put_model(m)
    dx0 = mjx.make_data(mx)

    # Per-env start pose (x, y, yaw) and random wheel commands so rays differ.
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    xy = jax.random.uniform(k1, (num_envs, 2), minval=-0.5, maxval=0.5)
    yaw = jax.random.uniform(k2, (num_envs,), minval=-np.pi, maxval=np.pi)
    ctrl = jax.random.uniform(k3, (num_envs, 2), minval=-arena.MAX_WHEEL_SPEED,
                              maxval=arena.MAX_WHEEL_SPEED)

    def init(xy_i, yaw_i, ctrl_i):
        qpos = dx0.qpos.at[0:2].set(xy_i)
        qpos = qpos.at[3:7].set(jp.array([jp.cos(yaw_i / 2), 0.0, 0.0, jp.sin(yaw_i / 2)]))
        d = dx0.replace(qpos=qpos, ctrl=ctrl_i)
        return mjx.forward(mx, d)

    batch = jax.jit(jax.vmap(init))(xy, yaw, ctrl)

    step_v = jax.vmap(mjx.step, in_axes=(None, 0))

    @jax.jit
    def rollout(d):
        def body(d, _):
            d = step_v(mx, d)
            return d, d.sensordata
        return jax.lax.scan(body, d, None, length=n_steps)

    # Correctness pass: every step returns sensordata of shape (num_envs, n_rays + 1).
    d1, sens = rollout(batch)
    sens = np.asarray(jax.block_until_ready(sens))
    rays = sens[:, :, :n_rays]
    up = sens[:, :, n_rays]
    res["sensordata_shape_per_step"] = list(sens.shape[1:])
    cond["steps_on_gpu_at_count"] = bool(cond["gpu_backend"] and sens.shape[0] == n_steps
                                         and sens.shape[1] == num_envs)
    cond["range_rays_every_step"] = bool(rays.shape == (n_steps, num_envs, n_rays)
                                         and np.isfinite(rays).all())
    hits = rays[rays >= 0]
    res["ray_hit_range_m"] = [float(hits.min()), float(hits.max())] if hits.size else None
    res["ray_hit_fraction"] = float((rays >= 0).mean())
    res["miss_value_observed"] = sorted({float(v) for v in np.unique(up)})
    cond["miss_value_proven"] = bool(np.all(up == up.flat[0]) and up.flat[0] < 0)
    # Self-hit check: a reading shorter than the chassis radius would mean a ray hit the robot.
    cond["no_self_hits"] = bool(hits.size == 0 or hits.min() > 0.15)
    res["robot_z_range_m"] = [float(np.asarray(d1.qpos[:, 2]).min()), float(np.asarray(d1.qpos[:, 2]).max())]
    cond["physics_finite"] = bool(np.isfinite(np.asarray(d1.qpos)).all())

    # Throughput: repeat the jitted rollout after warm-up.
    reps = 3
    t0 = time.perf_counter()
    for _ in range(reps):
        d_out, s_out = rollout(batch)
        jax.block_until_ready(s_out)
    dt = time.perf_counter() - t0
    res["wall_s"] = dt
    res["env_steps_per_s"] = num_envs * n_steps * reps / dt

    # ONNX path: Brax PPO policy network (the network type Playground trains),
    # exported by building the graph from its weights, checked against ONNX Runtime.
    try:
        res["onnx"] = onnx_check(n_rays + 5)
        cond["onnx_export_path"] = res["onnx"]["max_abs_diff"] <= 1e-4
    except Exception as e:  # recorded, not fatal: tie-break only
        res["onnx"] = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]}
        cond["onnx_export_path"] = False
    return res


def onnx_check(obs_dim: int) -> dict:
    import jax
    import jax.numpy as jp
    import numpy as np
    import onnx
    import onnxruntime as ort
    from brax.training.agents.ppo import networks as ppo_networks
    from onnx import TensorProto, helper, numpy_helper

    from brax.training.acme import running_statistics

    # Brax PPO passes running_statistics.normalize when normalize_observations=True.
    nets = ppo_networks.make_ppo_networks(obs_dim, 2, policy_hidden_layer_sizes=(32, 32),
                                          preprocess_observations_fn=running_statistics.normalize)
    params = nets.policy_network.init(jax.random.PRNGKey(1))
    # Brax policy apply signature: (normalizer_params, policy_params, obs).
    norm = running_statistics.init_state(jax.ShapeDtypeStruct((obs_dim,), jp.float32))
    norm = norm.replace(mean=jp.linspace(-0.1, 0.1, obs_dim), std=jp.linspace(0.5, 1.5, obs_dim))
    obs = jax.random.uniform(jax.random.PRNGKey(2), (64, obs_dim), minval=-1.0, maxval=1.0)
    logits = nets.policy_network.apply(norm, params, obs)
    ref = np.asarray(jp.tanh(logits[:, :2]))  # NormalTanhDistribution mode

    # Build: (obs - mean) / std -> [Dense -> swish]* -> Dense -> slice loc -> tanh
    layers = [params["params"][k] for k in sorted(params["params"], key=lambda s: int(s.split("_")[-1]))]
    nodes, inits = [], []
    inits.append(numpy_helper.from_array(np.asarray(norm.mean, np.float32), "mean"))
    inits.append(numpy_helper.from_array(np.asarray(norm.std, np.float32), "std"))
    nodes += [helper.make_node("Sub", ["obs", "mean"], ["c"]), helper.make_node("Div", ["c", "std"], ["h0"])]
    h = "h0"
    for i, layer in enumerate(layers):
        inits.append(numpy_helper.from_array(np.asarray(layer["kernel"], np.float32), f"W{i}"))
        inits.append(numpy_helper.from_array(np.asarray(layer["bias"], np.float32), f"b{i}"))
        nodes.append(helper.make_node("Gemm", [h, f"W{i}", f"b{i}"], [f"z{i}"]))
        if i < len(layers) - 1:
            nodes += [helper.make_node("Sigmoid", [f"z{i}"], [f"s{i}"]),
                      helper.make_node("Mul", [f"z{i}", f"s{i}"], [f"h{i + 1}"])]
            h = f"h{i + 1}"
    inits += [numpy_helper.from_array(np.array([0], np.int64), "st"),
              numpy_helper.from_array(np.array([2], np.int64), "en"),
              numpy_helper.from_array(np.array([1], np.int64), "ax")]
    nodes += [helper.make_node("Slice", [f"z{len(layers) - 1}", "st", "en", "ax"], ["loc"]),
              helper.make_node("Tanh", ["loc"], ["action"])]
    graph = helper.make_graph(nodes, "policy", [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [None, obs_dim])],
                              [helper.make_tensor_value_info("action", TensorProto.FLOAT, [None, 2])], inits)
    # IR version 8 pairs with opset 17 and loads in older ONNX Runtime builds too.
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    onnx.checker.check_model(model)
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    out = sess.run(["action"], {"obs": np.asarray(obs, np.float32)})[0]
    return {"exporter": "onnx.helper graph from Brax PPO policy weights", "opset": 17,
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
        res = {"framework": "mjx", "error": f"{type(e).__name__}: {e}",
               "trace": traceback.format_exc()[-4000:], "conditions": {}, "versions": versions()}
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "trace"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
